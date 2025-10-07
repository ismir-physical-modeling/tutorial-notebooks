import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float, PRNGKeyArray


# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def apply_lru_dynamics_from_ic(
    ic: Float[Array, "T d_model"],
    n_steps: int,
    discrete_lambda: Float[Array, " d_hidden"],
    B_norm: Float[Array, "d_hidden d_model"],
    C: Float[Array, "d_model d_hidden"],
) -> Float[Array, "n_steps d_model"]:
    Lambda_elements = jnp.repeat(
        discrete_lambda[None, ...],
        n_steps - 1,
        axis=0,
    )

    # Add initial identity element for scan
    Lambda_elements = jnp.concatenate(
        [
            jnp.ones((1, discrete_lambda.shape[0]), dtype=jnp.complex64),
            Lambda_elements,
        ],
        axis=0,
    )

    h0 = B_norm @ ic[0]
    hidden_states = (
        jax.lax.associative_scan(
            jnp.multiply,
            Lambda_elements,
        )
        * h0
    )
    return jax.vmap(lambda h: jnp.real(C @ h))(hidden_states)


def apply_lru_dynamics(
    inputs: Float[Array, "T d_model"],
    discrete_lambda: Float[Array, " d_hidden"],
    B_norm: Float[Array, "d_hidden d_model"],
    C: Float[Array, "d_model d_hidden"],
    D: Float[Array, " d_model"],
):
    Lambda_elements = jnp.repeat(
        discrete_lambda[None, ...],
        inputs.shape[0] - 1,
        axis=0,
    )

    Lambda_elements = jnp.concatenate(
        [
            jnp.ones((1, discrete_lambda.shape[0]), dtype=jnp.complex64),
            Lambda_elements,
        ],
        axis=0,
    )

    Bu_elements = jax.vmap(lambda u: B_norm @ u)(inputs)
    _, hidden_states = jax.lax.associative_scan(
        binary_operator, (Lambda_elements, Bu_elements)
    )
    return jax.vmap(lambda h, x: jnp.real(C @ h) + D * x)(hidden_states, inputs)


def nu_init(
    key: PRNGKeyArray, shape: tuple, r_min: float, r_max: float
) -> Float[Array, "..."]:
    """Initialize radial parameters for LRU dynamics."""
    u = jax.random.uniform(key=key, shape=shape, dtype=jnp.float32)
    return jnp.log(-0.5 * jnp.log(u * (r_max**2 - r_min**2) + r_min**2))


def theta_init(
    key: PRNGKeyArray, shape: tuple, max_phase: float
) -> Float[Array, "..."]:
    """Initialize phase parameters for LRU dynamics."""
    u = jax.random.uniform(key, shape=shape, dtype=jnp.float32)
    return jnp.log(max_phase * u)


def matrix_init(
    key: PRNGKeyArray,
    shape: tuple,
    dtype=jnp.float32,
    normalization: float = 1.0,
) -> Float[Array, "..."]:
    return jax.random.normal(key=key, shape=shape, dtype=dtype) / normalization


def gamma_log_init(
    key: PRNGKeyArray,
    nu_log: Float[Array, " d_hidden"],
    theta_log: Float[Array, " d_hidden"],
) -> Float[Array, " d_hidden"]:
    diag_lambda = jnp.exp(-jnp.exp(nu_log) + 1j * jnp.exp(theta_log))
    return jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))


class LRU(eqx.Module):
    """
    LRU module in charge of the recurrent processing.
    Implementation following the one of Orvieto et al. 2023.
    """

    # Parameters
    d_hidden: int
    d_model: int
    r_min: float
    r_max: float
    max_phase: float
    n_steps: int | None

    # Learned parameters
    theta_log: Float[Array, " d_hidden"]
    nu_log: Float[Array, " d_hidden"]
    gamma_log: Float[Array, " d_hidden"]
    B_re: Float[Array, "d_hidden d_model"]
    B_im: Float[Array, "d_hidden d_model"]
    C_re: Float[Array, "d_model d_hidden"]
    C_im: Float[Array, "d_model d_hidden"]
    D: Float[Array, " d_model"]

    def __init__(
        self,
        d_hidden: int,
        d_model: int,
        key: PRNGKeyArray,
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = 6.28,
        n_steps: int | None = None,
    ):
        self.d_hidden = d_hidden
        self.d_model = d_model
        self.r_min = r_min
        self.r_max = r_max
        self.max_phase = max_phase
        self.n_steps = n_steps

        keys = jax.random.split(key, 8)

        # Initialize parameters
        self.theta_log = theta_init(
            keys[0],
            (d_hidden,),
            max_phase,
        )
        self.nu_log = nu_init(
            keys[1],
            (d_hidden,),
            r_min,
            r_max,
        )
        self.gamma_log = gamma_log_init(
            keys[2],
            self.nu_log,
            self.theta_log,
        )

        # Glorot initialized Input/Output projection matrices
        self.B_re = matrix_init(
            keys[3], (d_hidden, d_model), normalization=float(jnp.sqrt(2 * d_model))
        )
        self.B_im = matrix_init(
            keys[4], (d_hidden, d_model), normalization=float(jnp.sqrt(2 * d_model))
        )
        self.C_re = matrix_init(
            keys[5], (d_model, d_hidden), normalization=float(jnp.sqrt(d_hidden))
        )
        self.C_im = matrix_init(
            keys[6], (d_model, d_hidden), normalization=float(jnp.sqrt(d_hidden))
        )
        self.D = matrix_init(keys[7], (d_model,))

    def __call__(
        self,
        inputs: Float[Array, "T d_model"],
    ) -> Float[Array, "T d_model"]:

        # Compute derived parameters
        C = self.C_re + 1j * self.C_im
        B = self.B_re + 1j * self.B_im
        B_norm = B * jnp.exp(self.gamma_log)[..., None]
        discrete_diag_lambda = jnp.exp(
            -jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log)
        )


        if self.n_steps is None:
            return apply_lru_dynamics(
                inputs,
                discrete_diag_lambda,
                B_norm,
                C,
                self.D,
            )
        else:
            return apply_lru_dynamics_from_ic(
                inputs,
                self.n_steps,
                discrete_diag_lambda,
                B_norm,
                C,
            )


class SequenceLayer(eqx.Module):
    """Single layer, with one SSM module, GLU, dropout and layer norm"""

    seq: LRU
    out1: eqx.nn.Linear
    out2: eqx.nn.Linear
    mlp: eqx.nn.MLP
    normalization: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    d_model: int
    activation: str
    prenorm: bool

    def __init__(
        self,
        ssm: LRU,
        d_model: int,
        key: PRNGKeyArray,
        dropout: float = 0.0,
        norm: str = "layer",
        activation: str = "half_glu1",
        prenorm: bool = True,
    ):
        self.seq = ssm
        self.d_model = d_model
        self.activation = activation
        self.prenorm = prenorm

        keys = jax.random.split(key, 3)

        # Output projections
        self.out1 = eqx.nn.Linear(d_model, d_model, key=keys[0])
        self.out2 = eqx.nn.Linear(d_model, d_model, key=keys[1])

        # MLP: d_model -> d_model * 4 -> d_model
        self.mlp = eqx.nn.MLP(
            in_size=d_model,
            out_size=d_model,
            width_size=d_model * 4,
            depth=2,
            activation=jax.nn.gelu,
            key=keys[2],
        )

        # Normalization
        if norm == "layer":
            self.normalization = eqx.nn.LayerNorm(d_model)
        else:
            raise NotImplementedError("Only layer norm is supported in Equinox version")

        # Dropout
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(
        self,
        x: Float[Array, "time d_model"],
        *,
        key: PRNGKeyArray,
    ) -> Float[Array, "time d_model"]:
        """Forward pass through the sequence layer.

        Args:
            x: Input tensor of shape (time, d_model)
            key: PRNG key for dropout

        Returns:
            Output tensor of shape (time, d_model)
        """
        skip = x

        if self.prenorm:
            x = jax.vmap(self.normalization)(x)  # pre normalization

        x = self.seq(x)  # call LRU

        # Split keys for dropout

        # Apply activation and dropout based on activation type
        if self.activation == "full_glu":
            x = jax.nn.gelu(x)
            x = jax.vmap(self.out1)(x) * jax.nn.sigmoid(jax.vmap(self.out2)(x))
        elif self.activation == "half_glu1":
            x = jax.nn.gelu(x)
            x = x * jax.nn.sigmoid(jax.vmap(self.out2)(x))
        elif self.activation == "gelu":
            x = jax.nn.gelu(x)
        elif self.activation == "mlp":
            x = jax.vmap(self.mlp)(x)
        else:
            raise NotImplementedError(f"Activation {self.activation} not implemented")

        x = skip + x  # skip connection

        if not self.prenorm:
            x = jax.vmap(self.normalization)(x)  # post normalization

        return x


class StackedSSM(eqx.Module):
    """Stacked SSM with multiple sequence layers."""

    first_layer: LRU | None
    layers: list[SequenceLayer]
    d_model: int
    d_vars: int
    n_steps: int | None

    def __init__(
        self,
        d_model: int,
        d_vars: int,
        n_layers: int,
        key: PRNGKeyArray,
        d_hidden: int | None = None,
        ssm_first_layer: bool = False,
        n_steps: int | None = None,
        dropout: float = 0.0,
        norm: str = "layer",
        activation: str = "half_glu1",
        prenorm: bool = True,
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = 6.28,
    ):
        self.d_model = d_model
        self.d_vars = d_vars
        self.n_steps = n_steps

        d_total = d_model * d_vars
        if d_hidden is None:
            d_hidden = d_total

        keys = jax.random.split(key, n_layers + 2)

        # Optional first layer for one-to-many prediction
        if ssm_first_layer:
            self.first_layer = LRU(
                d_hidden=d_hidden,
                d_model=d_total,
                key=keys[0],
                n_steps=n_steps,
                r_min=r_min,
                r_max=r_max,
                max_phase=max_phase,
            )
        else:
            self.first_layer = None

        # Create stacked layers
        self.layers = []
        for i in range(n_layers):
            # Create SSM module for this layer
            ssm = LRU(
                d_hidden=d_hidden,
                d_model=d_total,
                key=keys[i + 1],
                r_min=r_min,
                r_max=r_max,
                max_phase=max_phase,
            )

            # Create sequence layer
            layer = SequenceLayer(
                ssm=ssm,
                d_model=d_total,
                key=keys[i + 1 + n_layers],
                dropout=dropout,
                norm=norm,
                activation=activation,
                prenorm=prenorm,
            )
            self.layers.append(layer)

    def __call__(
        self,
        x: Float[Array, "T W C"],
        *,
        key: PRNGKeyArray,
    ) -> Float[Array, "T W C"]:
        # Flatten spatial and channel dimensions
        x = rearrange(x, "t w c -> t (w c)")

        # Apply first layer or initialize with zeros
        if self.first_layer is not None:
            x = self.first_layer(x)
        else:
            # One-to-many: expand first timestep to full sequence
            x = jnp.concatenate(
                [
                    x[0:1],
                    jnp.zeros((x.shape[0] - 1, x.shape[1])),
                ],
                axis=0,
            )

        # Apply stacked layers
        for i, layer in enumerate(self.layers):
            key_i = jax.random.fold_in(key, i)
            x = layer(x, key=key_i)

        # Reshape back to original dimensions
        return rearrange(x, "t (w c) -> t w c", w=self.d_model, c=self.d_vars)
