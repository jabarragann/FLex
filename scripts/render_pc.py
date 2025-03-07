import json
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

from JuanFlex.SimpleStereoMIS import LazyLoaderStereoMIS

from flex.model.HexPlane import HexPlane


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
    # transforms_train = transforms_path / "transforms_train.json"
    transforms_test = transforms_path / "transforms_test.json"

    with open(transforms_test) as test_json_file:
        test_json_dict = json.load(test_json_file)

    # with open(transforms_train) as train_json_file:
    #     train_json_dict = json.load(train_json_file)

    test_dataset = LazyLoaderStereoMIS(test_json_dict, transforms_path)
    sample = test_dataset[20]

    rays = sample.rays
    cur_time = sample.cur_time

    rays = rays.to(device)
    cur_time = cur_time.to(device)

    # render
    # camera_model = test_dataset.camera
    # batch_size = 8192
    # full_rgb_uint = render_full_image(
    #     model, rays, cur_time, camera_model.w, camera_model.h, batch_size=8142
    # )

    # save image
    iio.imwrite("./test_outputs/output_pc.png", sample.rgb)
    print("Image saved to output.png")


main.add_command(single_render)

if __name__ == "__main__":
    main()
