import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import click
import cv2
import imageio
import numpy as np
from tqdm import tqdm
from VideoUtils import VideoOpenCV, frame_calculation


@dataclass
class MyContext:
    input_video: Path
    output_dir: Path
    save_frames: bool
    start_ts: str
    end_ts: str
    downsample_factor: int


def split_frame(frame: np.ndarray):
    W, H = frame.shape[1], frame.shape[0]
    left = frame[0 : H // 2, 0:W]
    right = frame[H // 2 :, 0:W]

    return left, right


def video_to_frames(
    ctx: MyContext, max_res: int = 0, target_fps: int = -1, process_length: int = -1
):
    # Path to the input video
    video_path = Path(ctx.input_video)
    video_cv2 = VideoOpenCV(video_path)

    # Output directories for left and right images
    output_dir = video_path.parent / ctx.output_dir
    output_dir.mkdir(exist_ok=True)

    # Save context
    args_file = output_dir / "arguments.txt"
    with open(args_file, "w") as f:
        for key, value in asdict(ctx).items():
            f.write(f"{key}: {value}\n")
    print(f"Saved arguments to {args_file}")

    # Left and right directories
    left_dir = Path(".")
    right_dir = Path(".")
    if ctx.save_frames:
        left_dir = output_dir / "left_frames"
        right_dir = output_dir / "right_frames"
        left_dir.mkdir(exist_ok=True)
        right_dir.mkdir(exist_ok=True)

    left_video_path = output_dir / "left_video.mp4"
    left_writer = imageio.get_writer(left_video_path, fps=30, codec="libx264")

    # Set frames
    frame_count = 0
    save_count = 0
    start_frame, end_frame = frame_calculation(ctx.start_ts, ctx.end_ts, video_cv2)
    video_cv2.set_start_frame(start_frame)

    print(f"Start frame: {start_frame}, End frame: {end_frame}")

    with tqdm(
        total=video_cv2.total_frames, desc="Processing Video", unit="frame"
    ) as pbar:
        while frame_count <= end_frame and video_cv2.is_opened():
            ret, frame = video_cv2.get_frame()
            if not ret:
                break

            if frame_count == 0:
                print(frame.shape)
                cv2.imwrite(str(output_dir / "sample_interlaced_img.png"), frame)

            left_image, right_image = split_frame(frame)

            # Save extracted frames
            if ctx.save_frames:
                if (frame_count - start_frame) % ctx.downsample_factor == 0:
                    left_filename = left_dir / f"left_{frame_count:04d}.png"
                    right_filename = right_dir / f"right_{frame_count:04d}.png"
                    cv2.imwrite(str(left_filename), left_image)
                    cv2.imwrite(str(right_filename), right_image)
                    save_count += 1

            # Write frames to video
            left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
            left_writer.append_data(left_image)

            frame_count += 1
            pbar.update(1)

    left_writer.close()
    print(f"Processing complete. Extracted {frame_count} frames.")
    print(f"Saved frames to {output_dir}")
    print(f"Saved {save_count} frames.")


@click.group()
@click.option("--input_video", type=click.Path(), required=True)
@click.option("--output_dir", type=click.Path())
@click.option("-l", "--save_frames", is_flag=True, help="Save frames")
@click.option(
    "-s", "--start_ts", default="00:00:00", help="Start time in HH:MM:SS format"
)
@click.option("-e", "--end_ts", default="00:00:00", help="End time in HH:MM:SS format")
@click.option("-d", "--downsample_factor", default=1, help="Downsample factor")
@click.pass_context
def cli(
    ctx: click.Context,
    input_video: str,
    output_dir: bool,
    save_frames: bool,
    start_ts: str,
    end_ts: str,
    downsample_factor: int,
):
    input_video = Path(input_video) if input_video is not None else None
    output_dir = Path(output_dir) if output_dir is not None else None

    ctx.obj = MyContext(
        input_video=input_video,
        output_dir=output_dir,
        save_frames=save_frames,
        start_ts=start_ts,
        end_ts=end_ts,
        downsample_factor=downsample_factor,
    )


@cli.command()
@click.pass_obj
def run(ctx: MyContext):
    print(f"Loading video {ctx.input_video}...")

    if not ctx.input_video.exists():
        raise Exception(f"Error: {ctx.input_video} does not exist.")

    if ctx.output_dir is None:
        ctx.output_dir = ctx.input_video.parent / "left"
        ctx.output_dir.mkdir(exist_ok=True)
    else:
        ctx.output_dir.mkdir(exist_ok=True)

    video_to_frames(ctx)


# Entry point
if __name__ == "__main__":
    """
    run with

    python video_to_frames.py --input-video . --output-dir ../../ run

    see:
    https://click.palletsprojects.com/en/stable/api/#context

    or chat GPT chat.
    https://chatgpt.com/share/67f7be66-f0f4-8000-a4c7-b70dac9a91c9
    """
    cli()
