import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import click
import torch

sys.path.append(os.getcwd())
print(sys.path)

from flex.dataloader.ray_utils import get_ray_directions_blender


@dataclass
class CameraModel:
    w: int
    h: int
    cx: float
    cy: float
    focal: float

    def __post_init__(self):
        self.calculate_camera_rays()

    def calculate_camera_rays(self):
        self.directions = get_ray_directions_blender(
            self.h,
            self.w,
            [self.focal, self.focal],
            [
                self.cx,
                self.cy,
            ],  # test cx and cy instead of standard center TODO: remove if experiment failed
        )  # (h, w, 3)

        self.directions = self.directions / torch.norm(
            self.directions, dim=-1, keepdim=True
        )


def load_camera_poses(transforms_path: Path, downsample: float = 1.0):
    with open(transforms_path) as f:
        poses_info = json.load(f)

    # fix img_wh dim:
    w, h = poses_info["w"], poses_info["h"]
    w, h = int(w), int(h)
    cx = poses_info["cx"] / downsample
    cy = poses_info["cy"] / downsample
    focal = poses_info["fl_x"] / downsample
    camera_model = CameraModel(w, h, cx, cy, focal)

    print(camera_model.directions.shape)


@click.command()
@click.option("-p", "--model_path", "model_path", help="Path to the model file")
def main(model_path: str):
    # load model
    model = torch.load(model_path, weights_only=False)
    model.eval()

    # Load camera model
    transforms_path = Path("/home/juan95/JuanData/StereoMIS_FLex/P2_8_1/")
    transforms_train = transforms_path / "transforms_train.json"
    transforms_test = transforms_path / "transforms_test.json"

    load_camera_poses(transforms_train)
    pass
    # # render
    # with torch.no_grad():
    #     img = model.render()
    # # save image
    # img.save("output.png")
    # print("Image saved to output.png")


if __name__ == "__main__":
    main()
