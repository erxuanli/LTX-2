"""
Compute reference videos for IC-LoRA training.
This script provides a command-line interface for generating reference videos to be used for IC-LoRA training.
Note that it reads and writes to the same file (the output of caption_videos.py),
where it adds the "reference_path" field to the JSON.
Basic usage:
    # Compute reference videos for all videos in a directory
    compute_reference.py videos_dir/ --output videos_dir/captions.json
"""

# Standard library imports
import json
from pathlib import Path
from typing import Dict

# Third-party imports
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF  # noqa: N812
import typer

try:
    import OpenImageIO as oiio

    HAS_OIIO = True
except ImportError:
    HAS_OIIO = False
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from transformers.utils.logging import disable_progress_bar

# Local imports
from ltx_trainer.video_utils import read_video, save_video

# Initialize console and disable progress bars
console = Console()
disable_progress_bar()


def _is_exr_sequence_dir(path: Path) -> bool:
    """Return True if path is a directory containing .exr files."""
    return path.is_dir() and any(path.glob("*.exr"))


def _load_exr_sequence_as_sdr(path: Path) -> tuple[torch.Tensor, float]:
    """Load an EXR frame sequence and tone-map to SDR via gamma-2.2.

    Expects scene-linear values already in [0, 1] (as produced by the
    S2R-HDR preprocessing step).  Applies out = linear^(1/2.2) and
    returns a float tensor in [0, 1].

    Args:
        path: Directory containing sequentially-named .exr files.

    Returns:
        Tuple of (tensor [F, 3, H, W] float32 in [0, 1], fps).
    """
    if not HAS_OIIO:
        raise ImportError("EXR support requires OpenImageIO: pip install openimageio")

    exr_files = sorted(path.glob("*.exr"))
    if not exr_files:
        raise ValueError(f"No .exr files found in {path}")

    fps = 24.0
    fps_file = path / "fps.txt"
    if fps_file.exists():
        try:
            fps = float(fps_file.read_text().strip())
        except ValueError:
            pass

    frames = []
    for exr_path in exr_files:
        inp = oiio.ImageInput.open(str(exr_path))
        if inp is None:
            raise ValueError(f"OpenImageIO could not open {exr_path}: {oiio.geterror()}")
        spec = inp.spec()
        pixels = inp.read_image(oiio.FLOAT)  # [H, W, nchannels] float32
        inp.close()

        if pixels is None:
            raise ValueError(f"Failed to read pixel data from {exr_path}")

        channel_names = [spec.channelnames[i] for i in range(spec.nchannels)]
        r_idx = channel_names.index("R")
        g_idx = channel_names.index("G")
        b_idx = channel_names.index("B")
        frame = pixels[:, :, [r_idx, g_idx, b_idx]]  # [H, W, 3]

        # Gamma-2.2 tone-map: linear [0,1] -> display [0,1]
        frame = np.clip(frame, 0.0, 1.0)
        frame = np.power(frame, 1.0 / 2.2).astype(np.float32)

        frames.append(torch.from_numpy(frame).permute(2, 0, 1))  # [3, H, W]

    return torch.stack(frames, dim=0), fps  # [F, 3, H, W]


def compute_reference(
    images: torch.Tensor,
) -> torch.Tensor:
    """Compute Canny edge detection on a batch of images.
    Args:
        images: Batch of images tensor of shape [B, C, H, W]
    Returns:
        Binary edge masks tensor of shape [B, H, W]
    """
    # Convert to grayscale if needed
    if images.shape[1] == 3:
        images = TF.rgb_to_grayscale(images)

    # Ensure images are in [0, 1] range
    if images.max() > 1.0:
        images = images / 255.0

    # Compute Canny edges
    edge_masks = []
    for image in images:
        # Convert to numpy for OpenCV
        image_np = (image.squeeze().cpu().numpy() * 255).astype("uint8")

        # Apply Canny edge detection
        edges = cv2.Canny(
            image_np,
            threshold1=100,
            threshold2=200,
        )

        # Convert back to tensor
        edge_mask = torch.from_numpy(edges).float()
        edge_masks.append(edge_mask)

    edges = torch.stack(edge_masks)
    edges = torch.stack([edges] * 3, dim=1)  # Convert to 3-channel
    return edges


def _get_meta_data(
    output_path: Path,
) -> Dict[str, str]:
    """Get set of existing reference video paths without loading the actual files.
    Args:
        output_path: Path to the reference video paths file
    Returns:
        Dictionary mapping media paths to reference video paths
    """
    if not output_path.exists():
        return {}

    console.print(f"[bold blue]Reading meta data from [cyan]{output_path}[/]...[/]")

    try:
        with output_path.open("r", encoding="utf-8") as f:
            json_data = json.load(f)
        return json_data

    except Exception as e:
        console.print(f"[bold yellow]Warning: Could not check meta data: {e}[/]")
        return {}


def _save_dataset_json(
    reference_paths: Dict[str, str],
    output_path: Path,
) -> None:
    """Save dataset json with reference video paths.
    Args:
        reference_paths: Dictionary mapping media paths to reference video paths
        output_path: Path to save the output file
    """

    with output_path.open("r", encoding="utf-8") as f:
        json_data = json.load(f)
        new_json_data = json_data.copy()
        for i, item in enumerate(json_data):
            media_path = item["media_path"]
            reference_path = reference_paths[media_path]
            new_json_data[i]["reference_path"] = reference_path

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(new_json_data, f, indent=2, ensure_ascii=False)

    console.print(f"[bold green]✓[/] Reference video paths saved to [cyan]{output_path}[/]")
    console.print("[bold yellow]Note:[/] Use these files with ImageOrVideoDataset by setting:")
    console.print("  reference_column='[cyan]reference_path[/]'")
    console.print("  video_column='[cyan]media_path[/]'")


def process_media(
    input_path: Path,
    output_path: Path,
    override: bool,
    batch_size: int = 100,
) -> None:
    """Process videos and images to compute condition on videos.
    Args:
        input_path: Path to input video/image file or directory
        output_path: Path to output reference video file
        override: Whether to override existing reference video files
    """
    if not output_path.exists():
        raise FileNotFoundError(
            f"Output file does not exist: {output_path}. This is also the input file for the dataset."
        )

    # Check for existing reference video files
    meta_data = _get_meta_data(output_path)

    base_dir = input_path.resolve()
    console.print(f"Using [bold blue]{base_dir}[/] as base directory for relative paths")

    # Filter media files
    media_to_process = []
    skipped_media = []

    def media_path_to_reference_path(media_file: Path) -> Path:
        # EXR sequence directories have no file extension; save their SDR
        # reference as an MP4 alongside the directory.
        if not media_file.suffix:
            return media_file.parent / (media_file.stem + "_sdr.mp4")
        return media_file.parent / (media_file.stem + "_reference" + media_file.suffix)

    media_files = [base_dir / Path(sample["media_path"]) for sample in meta_data]
    for media_file in media_files:
        reference_path = media_path_to_reference_path(media_file)
        media_to_process.append(media_file)

    console.print(f"Processing [bold]{len(media_to_process)}[/] media.")

    # Initialize progress tracking
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    )

    # Process media files
    media_paths = [item["media_path"] for item in meta_data]
    reference_paths = {rel_path: str(media_path_to_reference_path(Path(rel_path))) for rel_path in media_paths}

    with progress:
        task = progress.add_task("Computing condition on videos", total=len(media_to_process))

        for media_file in media_to_process:
            progress.update(task, description=f"Processing [bold blue]{media_file.name}[/]")

            rel_path = str(media_file.resolve().relative_to(base_dir))
            reference_path = media_path_to_reference_path(media_file)
            reference_paths[rel_path] = str(reference_path.relative_to(base_dir))

            if not reference_path.resolve().exists() or override:
                try:
                    if _is_exr_sequence_dir(media_file):
                        # S2R-HDR path: generate SDR reference via gamma-2.2 tone-mapping
                        all_condition, fps = _load_exr_sequence_as_sdr(media_file)
                    else:
                        # Default path: Canny edge detection on regular video
                        video, fps = read_video(media_file)
                        condition_frames = []
                        for i in range(0, len(video), batch_size):
                            batch = video[i : i + batch_size]
                            condition_frames.append(compute_reference(batch))
                        all_condition = torch.cat(condition_frames, dim=0)

                    save_video(all_condition, reference_path.resolve(), fps=fps)

                except Exception as e:
                    console.print(f"[bold red]Error processing [bold blue]{media_file}[/]: {e}[/]")
                    reference_paths.pop(rel_path)
            else:
                skipped_media.append(media_file)

            progress.advance(task)

    # Save results
    _save_dataset_json(reference_paths, output_path)

    # Print summary
    total_to_process = len(media_files) - len(skipped_media)
    console.print(
        f"[bold green]✓[/] Processed [bold]{total_to_process}/{len(media_files)}[/] media successfully.",
    )


app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Compute reference videos for IC-LoRA training.",
)


@app.command()
def main(
    input_path: Path = typer.Argument(  # noqa: B008
        ...,
        help="Path to input video/image file or directory containing media files",
        exists=True,
    ),
    output: Path | None = typer.Option(  # noqa: B008
        None,
        "--output",
        "-o",
        help="Path to json output file for reference video paths. "
        "This is also the input file for the dataset, the output of compute_captions.py.",
    ),
    override: bool = typer.Option(
        False,
        "--override",
        help="Whether to override existing reference video files",
    ),
    batch_size: int = typer.Option(
        100,
        "--batch-size",
        help="Batch size for processing videos",
    ),
) -> None:
    """Compute reference videos for IC-LoRA training.
    This script generates reference videos (e.g., Canny edge maps) for given videos.
    The paths in the output file will be relative to the output file's directory.
    Examples:
        # Process all videos in a directory
        compute_reference.py videos_dir/ -o videos_dir/captions.json
    """

    # Ensure output path is absolute
    output = Path(output).resolve()
    console.print(f"Output will be saved to [bold blue]{output}[/]")

    # Verify output path exists
    if not output.exists():
        raise FileNotFoundError(f"Output file does not exist: {output}. This is also the input file for the dataset.")

    # Process media files
    process_media(
        input_path=input_path,
        output_path=output,
        override=override,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    app()
