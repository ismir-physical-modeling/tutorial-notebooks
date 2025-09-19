import math
import os
from pathlib import Path
from typing import Any

import equinox as eqx
import ffmpeg
import gdown
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax_dataloader import DataLoader, Dataset
from jaxtyping import Array, PRNGKeyArray


def download_data(
    url: str = "https://drive.google.com/file/d/1fMNA7Gx04UOBXDM4JL1QStzqO85hpfmo/view?usp=sharing",
    output: str = "data/string_nonlin_100_Gaussian_16000Hz_1.0s.npy",
) -> None:
    """Download data from Google Drive using gdown."""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False, fuzzy=True)


def trunc_init(key: PRNGKeyArray, shape: tuple, dtype=jnp.float32) -> Array:
    """Truncated normal initialization for neural network weights."""
    out, in_ = shape
    stddev = math.sqrt(1 / in_)
    return stddev * jax.random.truncated_normal(
        key,
        shape=shape,
        lower=-2,
        upper=2,
        dtype=dtype,
    )


def init_linear_weight(model: eqx.Module, init_fn, key):
    """
    Initialize weights in a model using a specified initialization function.

    Args:
        model: PyTree containing eqx.nn.Linear layers
        init_fn: JAX initializer function with signature (key, shape, dtype)
        key: PRNG key
    """

    def is_linear(x):
        return isinstance(x, eqx.nn.Linear)

    def get_weights(m):
        return [
            x.weight
            for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
            if is_linear(x)
        ]

    weights = get_weights(model)
    if not weights:
        return model

    keys = jax.random.split(key, len(weights))
    new_weights = [
        init_fn(subkey, weight.shape, weight.dtype)
        for weight, subkey in zip(weights, keys)
    ]

    new_model = eqx.tree_at(get_weights, model, new_weights)
    return new_model


def standardize(x, mean, std, only_scale=True):
    return (x - mean) / std if not only_scale else x / std


def split_data(
    mapped_data: Array,
    split: list[float],
    extract_channels: list[int],
) -> tuple[Array, Array, Array]:
    n_train, n_val, n_test = [
        int(fraction * mapped_data.shape[0]) for fraction in split
    ]

    train = mapped_data[:n_train]
    val = mapped_data[n_train : n_train + n_val]
    test = mapped_data[n_train + n_val : n_train + n_val + n_test]

    if extract_channels:
        train = train[..., extract_channels]
        val = val[..., extract_channels]
        test = test[..., extract_channels]

    return train, val, test


class ArrayDataset(Dataset):
    def __init__(
        self,
        array: Array,
    ):
        self.array = array

    def __len__(self):
        return self.array.shape[0]

    def __getitem__(self, index):
        return self.array[index]


def load_and_preprocess_data(
    data_array_path: Path | str,
    split: list[float] = [0.8, 0.1, 0.1],
    extract_channels: list[int] = [0, 1],
    batch_size: int = 64,
    n_steps_train: int = 100,
    n_steps_val: int = 200,
    n_steps_test: int = 200,
):
    """Load trajectory data and preprocess for training.

    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader)
    """
    mapped_data = np.load(
        data_array_path,
        mmap_mode="r",
    )

    train_array, val_array, test_array = split_data(
        mapped_data,
        split=split,
        extract_channels=extract_channels,
    )

    mean = np.mean(train_array, axis=(0, 1, 2))
    # here we just use max for scaling instead of std
    # so that both position and velocity are scaled similarly
    max_per_dim = np.max(np.abs(train_array), axis=(0, 1, 2))

    train_array = standardize(train_array, mean, max_per_dim)
    val_array = standardize(val_array, mean, max_per_dim)
    test_array = standardize(test_array, mean, max_per_dim)

    train_array = train_array[:, :n_steps_train, ...]
    val_array = val_array[:, :n_steps_val, ...]
    test_array = test_array[:, :n_steps_test, ...]

    print(f"Train shape: {train_array.shape}")
    print(f"Val shape: {val_array.shape}")
    print(f"Test shape: {test_array.shape}")

    train_dataloader = DataLoader(
        ArrayDataset(train_array),
        batch_size=batch_size,
        shuffle=True,
        backend="jax",
    )

    val_dataloader = DataLoader(
        ArrayDataset(val_array),
        batch_size=batch_size,
        shuffle=False,
        backend="jax",
    )

    test_dataloader = DataLoader(
        ArrayDataset(test_array),
        batch_size=batch_size,
        shuffle=False,
        backend="jax",
    )

    return (
        train_dataloader,
        val_dataloader,
        test_dataloader,
    )


def visualize_results(
    model: Any,
    test_dataloader: DataLoader,
    losses: Array,
    model_name: str = "Model",
    save_path: Path | None = None,
    show_plot: bool = True,
    n_steps_train: int | None = None,
    show_loss_plot: bool = True,
    signal_range: float | None = 1.0,
):
    """Visualize training results using colormesh plots for test trajectory comparison.

    Args:
        model: Trained model (FNO1D, KoopmanAutoencoder1D, etc.)
        test_dataloader: Test dataloader
        losses: Training losses array
        model_name: Name of the model for plot titles and default filename
        save_path: Path to save plot (if None, saves as '{model_name.lower()}_results.png')
        show_plot: Whether to display the plot
        n_steps_train: Number of training steps to mark boundary between training and test
        show_loss_plot: Whether to include the training loss subplot
        signal_range: Signal range for normalization in difference plot
    """
    test_sample = next(iter(test_dataloader))

    # Remove batch dim for single sample and readd later for consistency
    test_pred_sample = model(test_sample[0, 0:1])
    test_pred_sample = test_pred_sample[None]

    n_plots = 4 if show_loss_plot else 3
    fig, axes = plt.subplots(
        1,
        n_plots,
        figsize=(5 * n_plots, 5),
    )

    plot_offset = 0
    if show_loss_plot:
        losses_array = np.array(losses)
        valid_epochs = ~np.isnan(losses_array)
        valid_losses = losses_array[valid_epochs]
        epoch_indices = np.arange(len(losses_array))[valid_epochs]

        axes[0].semilogy(epoch_indices, valid_losses, "b-", linewidth=2)
        axes[0].set_title("Training Loss", fontsize=14)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("MSE Loss")
        axes[0].set_xlim(0, len(losses_array) - 1)
        axes[0].set_ylim(1e-2, 10)
        axes[0].grid(True)
        plot_offset = 1

    test_pred_2d = test_pred_sample[0, :, :, 0]
    test_target_2d = test_sample[0, : test_pred_2d.shape[0], :, 0]
    # Use provided signal_range or calculate from data if not provided
    if signal_range is None:
        signal_range = jnp.maximum(jnp.max(jnp.abs(test_target_2d)), 1e-8)
    test_diff_2d = jnp.abs(test_pred_2d - test_target_2d) / signal_range

    time_coords = jnp.arange(test_pred_2d.shape[0])
    spatial_coords = jnp.arange(test_pred_2d.shape[1])

    # Use ground truth min/max for consistent color scaling
    vmin = float(jnp.min(test_target_2d))
    vmax = float(jnp.max(test_target_2d))

    im1 = axes[plot_offset].pcolormesh(
        spatial_coords,
        time_coords,
        test_pred_2d,
        cmap="RdBu_r",
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )
    axes[plot_offset].set_title("Prediction", fontsize=14)
    axes[plot_offset].set_xlabel("Spatial Position")
    axes[plot_offset].set_ylabel("Time Steps")
    plt.colorbar(im1, ax=axes[plot_offset])

    im2 = axes[plot_offset + 1].pcolormesh(
        spatial_coords,
        time_coords,
        test_target_2d,
        cmap="RdBu_r",
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )
    axes[plot_offset + 1].set_title("Ground Truth", fontsize=14)
    axes[plot_offset + 1].set_xlabel("Spatial Position")
    axes[plot_offset + 1].set_ylabel("Time Steps")
    plt.colorbar(im2, ax=axes[plot_offset + 1])

    im3 = axes[plot_offset + 2].pcolormesh(
        spatial_coords,
        time_coords,
        test_diff_2d,
        cmap="viridis",
        shading="auto",
        vmin=0,
        vmax=1,
    )
    axes[plot_offset + 2].set_title("Rel. Absolute Difference", fontsize=14)
    axes[plot_offset + 2].set_xlabel("Spatial Position")
    axes[plot_offset + 2].set_ylabel("Time Steps")
    plt.colorbar(im3, ax=axes[plot_offset + 2])

    # Add horizontal line to mark training horizon boundary
    if n_steps_train is not None and n_steps_train < test_pred_2d.shape[0]:
        # Add line to all trajectory plots (skip loss plot if present)
        for ax in axes[plot_offset:]:
            ax.axhline(
                y=n_steps_train - 1,
                color="black",
                linestyle="--",
                linewidth=2,
                label="Training Horizon",
            )
        # Add legend only to the first trajectory plot
        axes[plot_offset].legend(loc="upper right")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        default_filename = f"{model_name.lower().replace(' ', '_')}_results.png"
        plt.savefig(default_filename, dpi=150, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    test_target_trimmed = test_sample[:, : test_pred_sample.shape[1], :, :]
    test_mse = jnp.mean((test_pred_sample - test_target_trimmed) ** 2)
    test_mae = jnp.mean(jnp.abs(test_pred_sample - test_target_trimmed))

    print(f"Test Sample - MSE: {test_mse:.6f}, MAE: {test_mae:.6f}")


def create_test_model(model: Any, n_steps_test: int) -> Any:
    """Create a test model with modified n_steps while keeping all trained weights.

    Args:
        model: Trained model (FNO1D, KoopmanAutoencoder1D, etc.)
        n_steps_test: Number of prediction steps for testing

    Returns:
        New model with updated n_steps parameter
    """
    return eqx.tree_at(lambda m: m.n_steps, model, n_steps_test)


# %%
def create_static_filter(
    model,
    static_params_lambda,
):
    # Create a filter where everything is False (trainable) by default
    is_static_filter = jax.tree_util.tree_map(lambda _: False, model)

    # Get the parameters selected by the lambda to determine how many True values we need
    selected_params = static_params_lambda(model)

    # Create tuple of True values matching the number of selected parameters
    if isinstance(selected_params, tuple):
        true_values = tuple(True for _ in selected_params)
    else:
        # Single parameter case
        true_values = True

    is_static_filter = eqx.tree_at(
        static_params_lambda,
        is_static_filter,
        true_values,
    )
    return is_static_filter


def render_animation_movie(
    input_dir: str | Path,
    output_file: str | Path | None = None,
    framerate: int = 10,
    file_pattern: str = "frame_%05d.png",
    output_format: str = "gif",
) -> None:
    """Render animation frames into a movie using ffmpeg.

    Args:
        input_dir: Directory containing the frame images
        output_file: Output movie file path. If None, uses input_dir name with appropriate extension
        framerate: Frames per second for the output movie
        file_pattern: Pattern for input frame files (e.g., "frame_%04d.png")
        output_format: Output format - "gif" or "mp4"

    Raises:
        FileNotFoundError: If input directory doesn't exist or no frames found
        ImportError: If ffmpeg-python is not installed
        RuntimeError: If ffmpeg execution fails
    """

    # Convert to Path objects
    input_dir = Path(input_dir)

    # Check if input directory exists
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input path is not a directory: {input_dir}")

    # Check if any frame files exist
    # Replace any %Nd pattern with * for globbing
    import re

    glob_pattern = re.sub(r"%\d+d", "*", file_pattern)
    frame_files = list(input_dir.glob(glob_pattern))
    if not frame_files:
        raise FileNotFoundError(
            f"No frame files found in {input_dir} matching pattern {file_pattern}"
        )

    print(f"Found {len(frame_files)} frame files in {input_dir}")

    # Generate output filename if not provided
    if output_file is None:
        output_file = input_dir.parent / f"{input_dir.name}_animation.{output_format}"
    else:
        output_file = Path(output_file)

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Build input pattern for ffmpeg
    input_pattern = str(input_dir / file_pattern)

    try:
        if output_format.lower() == "gif":
            # Fixed GIF generation with proper framerate handling
            cmd = ffmpeg.input(
                input_pattern,
                format="image2",
                start_number=1,
                framerate=framerate,  # Use desired framerate for input
            ).output(
                str(output_file),
                loop=0,
            )
            print(f"FFmpeg command: {' '.join(cmd.compile())}")
            process = cmd.run(overwrite_output=True, quiet=True)
        elif output_format.lower() == "webm":
            # WebM format with VP9 codec
            cmd = ffmpeg.input(
                input_pattern,
                format="image2",
                framerate=framerate,
                start_number=1,
            ).output(
                str(output_file),
                vcodec="libvpx-vp9",
                crf=30,  # Quality (0-63, lower = better quality)
                r=framerate,
                pix_fmt="yuv420p",
            )
            print(f"FFmpeg command: {' '.join(cmd.compile())}")
            process = cmd.run(overwrite_output=True, quiet=True)
        else:
            # For MP4 or other formats
            cmd = ffmpeg.input(
                input_pattern,
                format="image2",
                framerate=framerate,  # Use the desired framerate for input
                start_number=1,  # Start from frame 1, not 0
            ).output(
                str(output_file),
                vcodec="libx264",
                pix_fmt="yuv420p",
                r=framerate,  # output framerate (can be different from input)
                crf=18,  # quality (lower = better quality)
                vf="pad=ceil(iw/2)*2:ceil(ih/2)*2",  # Ensure dimensions are even
            )
            print(f"FFmpeg command: {' '.join(cmd.compile())}")
            process = cmd.run(overwrite_output=True, quiet=True)

        print(f"Animation saved to {output_file}")

    except ffmpeg.Error as e:
        raise RuntimeError(f"FFmpeg failed to create animation: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to create animation: {e}") from e


class ProgressPlotter:
    """Progress plotter for training visualization with automatic frame management."""

    def __init__(
        self,
        output_dir: str | Path = "tmp_training",
        model_name: str = "Model",
        framerate: int = 10,
        output_format: str = "webm",
    ):
        """Initialize progress plotter with directory setup and frame counter.

        Args:
            output_dir: Directory to save frames
            model_name: Name of the model for default animation filename
            framerate: Frames per second for animation
            output_format: Output format for animation (webm, gif, mp4)
        """
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.framerate = framerate
        self.output_format = output_format
        self.frame_counter = 0

        # Create directory and remove existing files
        if self.output_dir.exists():
            # Remove existing frame files
            for frame_file in self.output_dir.glob("frame_*.png"):
                frame_file.unlink()
        else:
            self.output_dir.mkdir(exist_ok=True)

        print(f"Progress plots will be saved to: {self.output_dir}")

    def __call__(
        self,
        model,
        test_dataloader,
        losses: Array,
        show_plot: bool = False,
        show_loss_plot: bool = False,
        signal_range: float | None = 1.0,
    ) -> Path:
        """Save a single training progress frame and return the saved path.

        Args:
            model: Trained model to visualize
            test_dataloader: Test dataloader for visualization
            losses: Array of training losses
            show_plot: Whether to display the plot
            show_loss_plot: Whether to show loss subplot
            signal_range: Signal range for normalization in difference plot (default: 1.0, None to auto-calculate)

        Returns:
            Path to the saved frame file
        """
        self.frame_counter += 1
        plot_path = self.output_dir / f"frame_{self.frame_counter:05d}.png"

        try:
            visualize_results(
                model,
                test_dataloader,
                losses,
                model_name=self.model_name,
                save_path=plot_path,
                show_plot=show_plot,
                show_loss_plot=show_loss_plot,
                signal_range=signal_range,
            )
            return plot_path
        except Exception as e:
            print(f"Warning: Failed to save frame {self.frame_counter}: {e}")
            # Return a dummy path to maintain frame numbering
            return plot_path

    def render_animation(
        self,
        output_file: str | Path | None = None,
    ) -> Path:
        """Render all saved frames into an animation movie.

        Args:
            output_file: Output filename (auto-generated if None)

        Returns:
            Path to the generated animation file
        """
        if output_file is None:
            model_name_clean = self.model_name.lower().replace(" ", "_")
            output_file = f"{model_name_clean}_training.{self.output_format}"

        output_path = Path(output_file)

        try:
            render_animation_movie(
                input_dir=self.output_dir,
                output_file=output_path,
                framerate=self.framerate,
                output_format=self.output_format,
            )
            print(f"Training animation saved as {output_path}")
            return output_path
        except Exception as e:
            print(f"Could not create animation: {e}")
            # Return the intended path even if creation failed
            return output_path

    def cleanup(self):
        """Clean up temporary directory and frame files."""
        if self.output_dir.exists():
            for frame_file in self.output_dir.glob("frame_*.png"):
                frame_file.unlink()
            try:
                self.output_dir.rmdir()
                print(f"Cleaned up directory: {self.output_dir}")
            except OSError:
                print(f"Directory not empty, keeping: {self.output_dir}")

    def save_pinn_frame(
        self,
        model,
        epoch: int,
        time_array: Array,
        true_solution: Array,
        training_data_time: Array,
        training_data_values: Array,
        true_params: dict,
        epoch_interval: int = 4000,
        title: str = "PINN Training Progress",
    ) -> Path:
        """Save a PINN training progress frame with predictions and parameter table.

        Args:
            model: PINN model with learnable parameters
            epoch: Current training epoch
            time_array: Full time array for predictions
            true_solution: True analytical solution
            training_data_time: Time points of training data
            training_data_values: Training data values
            true_params: Dictionary of true parameter values
            epoch_interval: Epoch interval for frame numbering
            title: Plot title

        Returns:
            Path to saved frame file
        """
        # Generate predictions
        y_pred = jax.vmap(model)(time_array)

        # Calculate frame index
        frame_idx = epoch // epoch_interval
        plot_path = self.output_dir / f"frame_{frame_idx:05d}.png"

        # Create figure
        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.4)

        # Time series comparison
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(
            time_array, true_solution, label="True Solution", linewidth=2, alpha=0.8
        )
        ax1.plot(
            training_data_time,
            training_data_values,
            "o",
            label="Training Data",
            markersize=4,
            alpha=0.7,
        )
        ax1.plot(
            time_array, y_pred, "--", label="PINN Prediction", linewidth=2, alpha=0.8
        )
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Displacement")
        ax1.set_title("PINN vs Analytical Solution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.8, 0.8)

        # Parameter table
        table_ax = fig.add_subplot(gs[1])
        table_ax.axis("off")

        # Build parameter table dynamically
        table_data = [["Parameter", "True", "Current", "Error (%)", "Epoch"]]

        # Add parameters based on model attributes
        for param_name, true_value in true_params.items():
            if hasattr(model, param_name):
                current_value = getattr(model, param_name).item()
                error_pct = abs(current_value - true_value) / abs(true_value) * 100
                table_data.append(
                    [
                        param_name,
                        f"{true_value:.3f}",
                        f"{current_value:.3f}",
                        f"{error_pct:.1f}%",
                        f"{epoch}" if param_name == list(true_params.keys())[0] else "",
                    ]
                )

        table = table_ax.table(
            cellText=table_data,
            cellLoc="center",
            loc="center",
            colWidths=[0.2, 0.15, 0.15, 0.15, 0.15],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header row
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor("#40466e")
            table[(0, i)].set_text_props(weight="bold", color="white")

        plt.suptitle(f"{title} - Epoch {epoch}", fontsize=14)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        return plot_path

    @property
    def frame_count(self) -> int:
        """Get the current number of saved frames."""
        return self.frame_counter
