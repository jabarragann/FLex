import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

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
    sample: int
    resolution: Tuple[int, int] = (640, 480)


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
    frame_idx = 0
    save_count = 0
    start_frame, end_frame = frame_calculation(ctx.start_ts, ctx.end_ts, video_cv2)

    print(f"Start frame: {start_frame}, End frame: {end_frame}")

    with tqdm(
        total=video_cv2.total_frames, desc="Processing Video", unit="frame"
    ) as pbar:
        while frame_idx <= end_frame and video_cv2.is_opened():
            ret, frame = video_cv2.get_frame()
            if not ret:
                break

            if frame_idx == 0:
                print(frame.shape)
                cv2.imwrite(str(output_dir / "sample_interlaced_img.png"), frame)

            left_image, right_image = split_frame(frame)

            # Save extracted frames
            if ctx.save_frames and frame_idx >= start_frame:
                if (frame_idx - start_frame) % ctx.sample == 0:
                    left_filename = left_dir / f"left_{frame_idx:06d}.png"
                    right_filename = right_dir / f"right_{frame_idx:06d}.png"

                    left_image_res = cv2.resize(left_image, ctx.resolution)
                    right_image_res = cv2.resize(right_image, ctx.resolution)

                    cv2.imwrite(str(left_filename), left_image_res)
                    cv2.imwrite(str(right_filename), right_image_res)
                    save_count += 1

            # Write frames to video
            left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
            left_writer.append_data(left_image)

            frame_idx += 1
            pbar.update(1)

    left_writer.close()
    print(f"Processing complete. Extracted {frame_idx} frames.")
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
@click.option("--resolution", default="640x512", help="Resolution of the saved images")
@click.pass_context
def cli(
    ctx: click.Context,
    input_video: str,
    output_dir: bool,
    save_frames: bool,
    start_ts: str,
    end_ts: str,
    downsample_factor: int,
    resolution: str,
):
    input_video = Path(input_video) if input_video is not None else None
    output_dir = Path(output_dir) if output_dir is not None else None

    resolution = resolution.split("x")
    res = (int(resolution[0]), int(resolution[1]))

    ctx.obj = MyContext(
        input_video=input_video,
        output_dir=output_dir,
        save_frames=save_frames,
        start_ts=start_ts,
        end_ts=end_ts,
        sample=downsample_factor,
        resolution=res,
    )


@cli.command()
@click.pass_obj
def run(ctx: MyContext):
    print(f"Loading video {ctx.input_video}...")
    print(f"Output directory: {ctx.output_dir}")

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

    ## Full command to run in bash script
    python JuanScripts/data_processing/StereoMis_video_to_frames.py \
        --input_video /home/juan95/JuanData/StereoMIS/P2_8/IFBS_ENDOSCOPE-part0008.mp4 \
        --output_dir /home/juan95/JuanData/StereoMIS_FLex_juan/P2_8_juan_clip_FLex_juan \
        -s 00:00:00 -e 00:00:18 --resolution 640x512 --save_frames \
        run

    see:
    https://click.palletsprojects.com/en/stable/api/#context

    or chat GPT chat.
    https://chatgpt.com/share/67f7be66-f0f4-8000-a4c7-b70dac9a91c9
    """
    cli()
