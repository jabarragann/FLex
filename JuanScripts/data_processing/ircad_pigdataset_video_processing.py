from __future__ import annotations

import os
import sys
from pathlib import Path

import click
import cv2
import imageio
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from VideoUtils import VideoOpenCV, frame_calculation, process_interlaced_frame


def process_video(
    context: click.Context,
    path: Path,
    output_dir: str,
    save_frames: bool,
    start_time: str = "00:00:00",
    end_time: str = "00:00:00",
    downsample_factor: int = 1,
):
    # Path to the input video
    video_path = Path(path)
    video_cv2 = VideoOpenCV(video_path)

    # Output directories for left and right images
    output_dir = video_path.parent / output_dir
    output_dir.mkdir(exist_ok=True)

    # Save context
    args_file = output_dir / "arguments.txt"
    with open(args_file, "w") as f:
        for key, value in context.params.items():
            f.write(f"{key}: {value}\n")
    print(f"Saved arguments to {args_file}")

    # Left and right directories
    left_dir = Path(".")
    right_dir = Path(".")
    if save_frames:
        left_dir = output_dir / "left_frames"
        right_dir = output_dir / "right_frames"
        left_dir.mkdir(exist_ok=True)
        right_dir.mkdir(exist_ok=True)

    left_video_path = output_dir / "left_video.mp4"
    left_writer = imageio.get_writer(left_video_path, fps=30, codec="libx264")

    # Set frames
    frame_count = 0
    save_count = 0
    start_frame, end_frame = frame_calculation(start_time, end_time, video_cv2)
    video_cv2.set_start_frame(start_frame)

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

            left_image, right_image = process_interlaced_frame(frame)

            # Save extracted frames
            if save_frames:
                if (frame_count - start_frame) % downsample_factor == 0:
                    left_filename = left_dir / f"left_{save_count:04d}.png"
                    right_filename = right_dir / f"right_{save_count:04d}.png"
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


@click.command()
@click.argument("video_path")
@click.argument("output_dir_name")
@click.option("-l", "--save_frames", is_flag=True, help="Save frames")
@click.option(
    "-s", "--start_ts", default="00:00:00", help="Start time in HH:MM:SS format"
)
@click.option("-e", "--end_ts", default="00:00:00", help="End time in HH:MM:SS format")
@click.option("-d", "--downsample_factor", default=1, help="Downsample factor")
@click.pass_context
def main(
    context,
    video_path: str,
    output_dir_name: str,
    save_frames: bool,
    start_ts: str,
    end_ts: str,
    downsample_factor: int,
):
    print(video_path)
    video_path: Path = Path(video_path)
    process_video(
        context,
        video_path,
        output_dir_name,
        save_frames,
        start_time=start_ts,
        end_time=end_ts,
        downsample_factor=downsample_factor,
    )


if __name__ == "__main__":
    main()
