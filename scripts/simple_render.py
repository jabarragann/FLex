import os
import sys
from pathlib import Path

import click
import imageio.v3 as iio
import numpy as np
import torch
from natsort import natsorted
from tqdm import tqdm

sys.path.append(os.getcwd())
# print(sys.path)

from JuanFlex.utils import load_data, render_full_image

from flex.model.HexPlane import HexPlane


########
# CLI
########
def model_path_option(func):
    return click.option(
        "-p",
        "--model-path",
        required=True,
        type=click.Path(exists=True),
        help="Path to the model file.",
    )(func)


@click.group()
def main():
    pass


@click.command()
@model_path_option
def single_render(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model
    model: HexPlane = torch.load(model_path, weights_only=False)
    model.eval()
    model.to(device)

    # Load camera model
    transforms_path = Path("/home/juan95/JuanData/StereoMIS_FLex/P2_8_1/")
    transforms_train = transforms_path / "transforms_train.json"
    transforms_test = transforms_path / "transforms_test.json"

    rays, cur_time, camera_model = load_data(20, transforms_train, transforms_test)
    rays = rays.to(device)
    cur_time = cur_time.to(device)

    # render
    # batch_size = 8192
    full_rgb_uint = render_full_image(
        model, rays, cur_time, camera_model.w, camera_model.h, batch_size=8142
    )

    # save image
    iio.imwrite("./test_outputs/output.png", full_rgb_uint)
    print("Image saved to output.png")


@click.command()
@model_path_option
def time_change_render(model_path: str):
    outpath = Path("./test_outputs/time_render/")
    if not outpath.exists():
        outpath.mkdir(parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model
    model: HexPlane = torch.load(model_path, weights_only=False)
    model.eval()
    model.to(device)

    # Load camera model
    transforms_path = Path("/home/juan95/JuanData/StereoMIS_FLex/P2_8_1/")
    transforms_train = transforms_path / "transforms_train.json"
    transforms_test = transforms_path / "transforms_test.json"

    rays, cur_time, camera_model = load_data(20, transforms_train, transforms_test)
    rays = rays.to(device)
    cur_time = cur_time.to(device)

    time_range = np.linspace(0.0, 0.030, 30)

    # Generate frames
    for idx, t in enumerate(tqdm(time_range, desc="Rendering time change")):
        cur_time = cur_time + t
        print(f"image time: {cur_time[0, 0]:.4f}")
        full_rgb_uint = render_full_image(
            model, rays, cur_time, camera_model.w, camera_model.h, batch_size=8192
        )
        # save image
        iio.imwrite(outpath / f"output_{idx:03d}.jpeg", full_rgb_uint)

    # Convert frames into video
    images = [str(img) for img in outpath.glob("*.jpeg")]
    images = natsorted(images)
    fps = 3
    with iio.imopen(outpath / "out.mp4", "w", plugin="pyav") as writer:
        writer.init_video_stream("libx264", fps=fps)  # Using H.264 codec
        for image_file in images:
            frame = iio.imread(image_file)  # Read the image
            writer.write_frame(frame)  # Write the frame

    print("Video saved to output.mp4")


main.add_command(single_render)
main.add_command(time_change_render)

if __name__ == "__main__":
    main()
