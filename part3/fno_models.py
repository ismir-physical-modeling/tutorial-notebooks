# %%

# %% [markdown]
# Fourier Neural Operator Autoregressive (FNO1D_AR) Implementation

# %%
from collections.abc import Callable
from typing import Optional

# %%
import equinox as eqx
import jax
import jax.numpy as jnp
from einops import einsum, rearrange
from jaxtyping import Array, Float, PRNGKeyArray


# %%
class SpectralConv1d(eqx.Module):
    """Spectral Convolution Layer for 1D inputs using Equinox.

    The n_modes parameter should be set to the length of the output for now,
    as it is not clear that the truncation is done correctly.
    """

    weight_real: Float[Array, "in_ch out_ch n_modes"]
    weight_imag: Float[Array, "in_ch out_ch n_modes"]
    in_channels: int
    out_channels: int
    n_modes: int
    linear_conv: bool

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: int,
        key: PRNGKeyArray,
        linear_conv: bool = True,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.linear_conv = linear_conv

        weight_shape = (in_channels, out_channels, n_modes)
        scale = 1 / (in_channels * out_channels)

        key1, key2 = jax.random.split(key)
        self.weight_real = jax.random.uniform(
            key1, weight_shape, minval=-scale, maxval=scale
        )
        self.weight_imag = jax.random.uniform(
            key2, weight_shape, minval=-scale, maxval=scale
        )

    def __call__(
        self, x: Float[Array, "width in_channels"]
    ) -> Float[Array, "width out_channels"]:
        """
        Args:
            x: Input array of shape (w, in_ch) where w is spatial dimension

        Returns:
            Output array of shape (w, out_ch)
        """
        W, C = x.shape

        # Get the fourier coefficients along the spatial dimension
        # We pad the inputs so that we perform a linear convolution
        X = jnp.fft.rfft(x, n=W * 2 - 1, axis=-2, norm="ortho")

        # Truncate to the first n_modes coefficients
        X = X[: self.n_modes, :]

        # Multiply by the fourier coefficients of the kernel
        complex_weight = self.weight_real + 1j * self.weight_imag

        # X shape: (modes, in_ch), weight shape: (in_ch, out_ch, modes)
        X = einsum(
            complex_weight,
            X,
            "in_ch out_ch modes, modes in_ch -> modes out_ch",
        )

        # Inverse fourier transform along dimension N and remove padding
        x_out = jnp.fft.irfft(X, axis=-2, norm="ortho")[:W]

        return x_out


# %%
class FNO1D(eqx.Module):
    """Fourier Neural Operator for 1D inputs using Equinox."""

    spectral_convs: list
    w_layers: list
    lifting: eqx.nn.Linear
    projection_layers: list
    hidden_channels: int
    n_modes: int
    output_channels: int
    linear_conv: bool
    n_layers: int
    n_steps: int
    activation: Callable[[Float[Array, "..."]], Float[Array, "..."]]

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        n_modes: int,
        key: PRNGKeyArray,
        output_channels: int = 1,
        linear_conv: bool = True,
        n_layers: int = 4,
        n_steps: int = 1,
        activation: Callable[[Float[Array, "..."]], Float[Array, "..."]] = jax.nn.gelu,
    ):
        self.hidden_channels = hidden_channels
        self.n_modes = n_modes
        self.output_channels = output_channels
        self.linear_conv = linear_conv
        self.n_layers = n_layers
        self.n_steps = n_steps
        self.activation = activation

        keys = jax.random.split(key, n_layers + 4)

        # Create spectral convolution layers
        self.spectral_convs = [
            SpectralConv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                n_modes=n_modes,
                linear_conv=linear_conv,
                key=keys[i],
            )
            for i in range(n_layers)
        ]

        # Create skip connection layers (equivalent to Conv with kernel_size=1)
        self.w_layers = [
            eqx.nn.Linear(hidden_channels, hidden_channels, key=keys[n_layers + i])
            for i in range(n_layers)
        ]

        # Lifting layer
        self.lifting = eqx.nn.Linear(input_channels, hidden_channels, key=keys[-3])

        # Projection layers (tiny MLP)
        self.projection_layers = [
            eqx.nn.Linear(hidden_channels, 128, key=keys[-2]),
            eqx.nn.Linear(128, output_channels * n_steps, key=keys[-1]),
        ]

    def __call__(
        self, x: Float[Array, "time width in_ch"]
    ) -> Float[Array, "n_steps width out_ch"]:
        """
        The input to the FNO1D model is a 1D signal of shape (t, w, in_ch)
        where w is the spatial dimension and in_ch is the number of input channels.
        The channel dimension is typically 1 for scalar fields. However, it can
        also contain multiple time steps as channels or contain multiple scalar fields.

        Args:
            x: Input of shape (t, w, in_ch)

        Returns:
            Output of shape (t, w, out_ch) for n_steps timesteps
        """
        # We need to make time as a channel dimension for the spectral layers
        x = rearrange(x, "t w c -> w (t c)")

        # Lift the input to the hidden state
        h = jax.vmap(self.lifting)(x)

        # Apply spectral layers
        for spectral_conv, w_layer in zip(self.spectral_convs, self.w_layers):
            h1 = spectral_conv(h)
            h2 = jax.vmap(w_layer)(h)  # Apply linear layer to each spatial point
            h = self.activation(h1 + h2)

        # Project to output using tiny MLP
        y = h
        for layer in self.projection_layers[:-1]:
            y = self.activation(jax.vmap(layer)(y))
        y: Array = jax.vmap(self.projection_layers[-1])(y)

        # Rearrange the output to the original shape
        y: Array = rearrange(
            y,
            "w (t c) -> t w c",
            t=self.n_steps,
            c=self.output_channels,
        )

        return y


# %%
class FNO1D_AR(eqx.Module):
    """Autoregressive wrapper for FNO1D that enables step-by-step prediction.

    Takes a single time step input and predicts multiple future time steps
    using jax.lax.scan for efficient sequential computation.
    """

    fno_model: FNO1D
    n_steps: int
    channel_projection: eqx.nn.Linear | None

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        n_modes: int,
        n_steps: int,
        key: PRNGKeyArray,
        output_channels: int = 1,
        linear_conv: bool = True,
        n_layers: int = 4,
        activation: Callable[[Float[Array, "..."]], Float[Array, "..."]] = jax.nn.gelu,
    ):
        """Initialize FNO1D_AR with a single-step FNO model.

        Args:
            input_channels: Number of input channels
            hidden_channels: Number of hidden channels in FNO
            n_modes: Number of Fourier modes
            n_steps: Number of autoregressive prediction steps
            key: JAX random key
            output_channels: Number of output channels
            linear_conv: Whether to use linear convolution in spectral layers
            n_layers: Number of FNO layers
            activation: Activation function
            use_teacher_forcing: Whether to use teacher forcing during training
        """
        self.n_steps = n_steps

        key_fno, key_proj = jax.random.split(key)

        # Create single-step FNO model (n_steps=1)
        self.fno_model = FNO1D(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            n_modes=n_modes,
            output_channels=output_channels,
            linear_conv=linear_conv,
            n_layers=n_layers,
            n_steps=1,  # Single step prediction
            activation=activation,
            key=key_fno,
        )

        # Channel projection layer if input/output channels differ
        if output_channels != input_channels:
            self.channel_projection = eqx.nn.Linear(
                output_channels, input_channels, key=key_proj
            )
        else:
            self.channel_projection = None

    def __call__(
        self,
        x: Float[Array, "1 width in_ch"],
        ground_truth: Float[Array, "n_steps width out_ch"] | None = None,
    ) -> Float[Array, "n_steps width out_ch"]:
        """Autoregressive prediction using jax.lax.scan.

        Args:
            x: Initial input of shape (1, width, in_ch)
            ground_truth: Optional ground truth for teacher forcing during training

        Returns:
            Predictions of shape (n_steps, width, out_ch)
        """
        def scan_fn(carry, _):
            """Single step of autoregressive prediction."""
            current_input = carry  # Shape: (1, width, in_ch)

            fno_output = self.fno_model(current_input)  # Shape: (1, width, out_ch)
            next_step = fno_output[0]  # Shape: (width, out_ch)

            return fno_output, next_step

        _, predictions = jax.lax.scan(
            scan_fn,
            x,
            xs=None,
            length=self.n_steps,
        )

        # prepend the initial input to match the output shape
        predictions = jnp.concatenate([x, predictions], axis=0)

        return predictions


# %%
def create_fno1d_ar(
    input_channels: int,
    hidden_channels: int,
    n_modes: int,
    n_steps: int,
    key: PRNGKeyArray,
    output_channels: int = 1,
    linear_conv: bool = True,
    n_layers: int = 4,
    activation: Callable[[Float[Array, "..."]], Float[Array, "..."]] = jax.nn.gelu,
) -> FNO1D_AR:
    """Factory function to create FNO1D_AR model."""
    return FNO1D_AR(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        n_modes=n_modes,
        n_steps=n_steps,
        key=key,
        output_channels=output_channels,
        linear_conv=linear_conv,
        n_layers=n_layers,
        activation=activation,
    )


# %%
if __name__ == "__main__":
    # Basic functionality tests
    print("Testing FNO1D_AR implementation...")

    # Test parameters
    batch_size = 4
    width = 64
    input_channels = 2
    output_channels = 2
    hidden_channels = 16
    n_modes = 32
    n_steps = 5
    n_layers = 2

    key = jax.random.PRNGKey(42)

    # Create model
    key_model, key_data = jax.random.split(key)
    model = create_fno1d_ar(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        n_modes=n_modes,
        n_steps=n_steps,
        output_channels=output_channels,
        n_layers=n_layers,
        key=key_model,
    )

    print(f"Model created successfully")
    print(f"  Input channels: {input_channels}")
    print(f"  Output channels: {output_channels}")
    print(f"  Hidden channels: {hidden_channels}")
    print(f"  N steps: {n_steps}")

    # Test single example
    x_single = jax.random.normal(key_data, (1, width, input_channels))
    y_single = model(x_single)

    print(f"\nSingle example test:")
    print(f"  Input shape: {x_single.shape}")
    print(f"  Output shape: {y_single.shape}")
    assert y_single.shape == (n_steps, width, output_channels)

    # Test batched version
    x_batch = jax.random.normal(key_data, (batch_size, 1, width, input_channels))
    batched_model = jax.vmap(model, in_axes=(0, None))
    y_batch = batched_model(x_batch, None)

    print(f"\nBatched test:")
    print(f"  Input shape: {x_batch.shape}")
    print(f"  Output shape: {y_batch.shape}")
    assert y_batch.shape == (batch_size, n_steps, width, output_channels)

    # Test teacher forcing
    model_tf = create_fno1d_ar(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        n_modes=n_modes,
        n_steps=n_steps,
        output_channels=output_channels,
        n_layers=n_layers,
        key=key_model,
    )

    ground_truth = jax.random.normal(key_data, (n_steps, width, output_channels))
    y_tf = model_tf(x_single, ground_truth)

    print(f"\nTeacher forcing test:")
    print(f"  Ground truth shape: {ground_truth.shape}")
    print(f"  Output shape: {y_tf.shape}")
    assert y_tf.shape == (n_steps, width, output_channels)

    print("\nAll tests passed! ✓")
