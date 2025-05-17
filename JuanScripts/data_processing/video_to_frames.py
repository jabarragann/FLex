from dataclasses import dataclass
from pathlib import Path

import click
import cv2


@dataclass
class MyContext:
    input_video: Path
    output_dir: Path


def video_to_frames(ctx: MyContext, max_res: int = 0, target_fps: int = -1):
    """Taken from video depth anything"""

    cap = cv2.VideoCapture(str(ctx.input_video))

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if max_res > 0 and max(original_height, original_width) > max_res:
        scale = max_res / max(original_height, original_width)
        height = round(original_height * scale)
        width = round(original_width * scale)

    fps = original_fps if target_fps < 0 else target_fps
    stride = max(round(original_fps / fps), 1)

    # frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if frame_count % stride == 0 and ret:
            if max_res > 0 and max(original_height, original_width) > max_res:
                frame = cv2.resize(frame, (width, height))  # Resize frame

            cv2.imwrite(
                str(ctx.output_dir / f"frame_{frame_count:06d}.png"),
                frame,
            )
        else:
            break

        frame_count += 1
        print("debug frame_count", frame_count)

    print(f"Extracted {frame_count} frames from {ctx.input_video}.")
    cap.release()


@click.group()
@click.option("--input-video", type=click.Path(), required=True)
@click.option("--output-dir", type=click.Path())
@click.pass_context
def cli(ctx: click.Context, input_video: str, output_dir: bool):
    ctx.obj = MyContext(input_video=Path(input_video), output_dir=output_dir)


# Step 3: Subcommand that uses the context with type hints
@cli.command()
@click.pass_obj
def run(ctx: MyContext):
    print(f"Loading video {ctx.input_video}...")

    if not ctx.input_video.exists():
        raise Exception(f"Error: {ctx.input_video} does not exist.")

    if ctx.output_dir is None:
        ctx.output_dir = ctx.input_video.parent / "frames"
        ctx.output_dir.mkdir(exist_ok=True)
    else:
        ctx.output_dir = Path(ctx.output_dir)
        ctx.output_dir.mkdir(exist_ok=True)

    video_to_frames(ctx)


# Entry point
if __name__ == "__main__":
    """
    run with

    python video_to_frames.py --input-video . --output-dir ../../ run

    see: https://click.palletsprojects.com/en/stable/api/#context
    """
    cli()
