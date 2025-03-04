import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import click
import imageio.v3 as iio
import numpy as np
import torch

sys.path.append(os.getcwd())
print(sys.path)

from flex.dataloader.ray_utils import get_ray_directions_blender, get_rays
from flex.model.HexPlane import HexPlane


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


def create_camera_model(poses_json: dict, downsample: float = 1.0) -> CameraModel:
    # fix img_wh dim:
    w, h = poses_json["w"], poses_json["h"]
    w, h = int(w), int(h)
    cx = poses_json["cx"] / downsample
    cy = poses_json["cy"] / downsample
    focal = poses_json["fl_x"] / downsample
    camera_model = CameraModel(w, h, cx, cy, focal)

    return camera_model


def load_data(train_transforms_path, test_transforms_path):
    with open(train_transforms_path) as f:
        train_poses_json = json.load(f)
    with open(test_transforms_path) as f:
        test_poses_json = json.load(f)

    camera_model = create_camera_model(train_poses_json)
    print(camera_model.directions.shape)

    # Load pose
    blender2opencv = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    frame = test_poses_json["frames"][40]
    pose = np.array(frame["transform_matrix"]) @ blender2opencv
    c2w = torch.FloatTensor(pose)

    # Create rays
    rays_o, rays_d = get_rays(camera_model.directions, c2w)  # Get rays, both (h*w, 3).
    rays = torch.cat([rays_o, rays_d], 1)

    # Load time
    cur_time = torch.tensor(frame["time"])
    cur_time = cur_time.expand(rays_o.shape[0], 1)
    time_scale = 1.0
    cur_time = time_scale * (cur_time * 2.0 - 1.0)

    print(rays_o.shape, rays_d.shape)
    print(rays.shape, cur_time.shape)

    return rays, cur_time, camera_model


@click.command()
@click.option("-p", "--model_path", "model_path", help="Path to the model file")
def main(model_path: str):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model
    model: HexPlane = torch.load(model_path, weights_only=False)
    model.eval()
    model.to(device)

    # Load camera model
    transforms_path = Path("/home/juan95/JuanData/StereoMIS_FLex/P2_8_1/")
    transforms_train = transforms_path / "transforms_train.json"
    transforms_test = transforms_path / "transforms_test.json"

    rays, cur_time, camera_model = load_data(transforms_train, transforms_test)
    rays = rays.to(device)
    cur_time = cur_time.to(device)

    # render
    batch_size = 4096 * 2
    full_rgb = []
    with torch.no_grad():
        for i in range(0, int(np.ceil(cur_time.shape[0] / batch_size))):
            start = i * batch_size
            end = start + batch_size
            rgb_map, depth_map, alpha, z_vals, xyz_sampled, weight = model.forward(
                rays[start:end, :],
                cur_time[start:end, :],
                white_bg=True,
                is_train=False,
                ndc_ray=False,
                N_samples=-1,
            )
            full_rgb.append(rgb_map)
            print(rgb_map.shape)

    full_rgb = torch.cat(full_rgb, 0)
    full_rgb = full_rgb.reshape(camera_model.h, camera_model.w, 3)

    print(full_rgb.shape)
    full_rgb_uint = (full_rgb * 255).cpu().numpy().astype(np.uint8)

    # save image
    iio.imwrite("output.png", full_rgb_uint)
    print("Image saved to output.png")


if __name__ == "__main__":
    main()
