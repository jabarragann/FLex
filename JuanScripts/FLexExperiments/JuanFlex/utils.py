import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.getcwd())
# print(sys.path)

from JuanFlex.Camera import create_camera_model

from flex.model.HexPlane import HexPlane


def load_data(test_idx, train_transforms_path, test_transforms_path):
    with open(train_transforms_path) as f:
        train_poses_json = json.load(f)
    with open(test_transforms_path) as f:
        test_poses_json = json.load(f)

    camera_model = create_camera_model(train_poses_json)
    print(camera_model.directions.shape)

    # Load pose
    frame = test_poses_json["frames"][test_idx]

    # Create rays
    pose_cv = np.array(frame["transform_matrix"])
    rays = camera_model.generate_rays_in_world_coord(pose_cv, fmt="cv2")

    # Load time
    cur_time = torch.tensor(frame["time"])
    cur_time = cur_time.expand(rays.shape[0], 1)
    time_scale = 1.0
    cur_time = time_scale * (cur_time * 2.0 - 1.0)

    # print(rays_o.shape, rays_d.shape)
    print(rays.shape, cur_time.shape)

    return rays, cur_time, camera_model


@torch.no_grad()
def render_full_image(
    model: HexPlane,
    rays: torch.Tensor,
    cur_time: torch.Tensor,
    width: int,
    height: int,
    batch_size: int = 8192,
) -> np.ndarray:
    # render
    full_rgb = []

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
        # print(rgb_map.shape)

    full_rgb = torch.cat(full_rgb, 0)
    full_rgb = full_rgb.reshape(height, width, 3)

    full_rgb_uint = (full_rgb * 255).cpu().numpy().astype(np.uint8)

    return full_rgb_uint
