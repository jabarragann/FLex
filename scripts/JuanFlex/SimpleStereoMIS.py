from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile as tiff
import torch
from JuanFlex.Camera import create_camera_model
from PIL import Image
from torch import nn


@dataclass
class Sample:
    rgb: torch.Tensor
    depth: torch.Tensor
    pose: torch.Tensor
    rays: torch.Tensor
    cur_time: torch.Tensor


@dataclass
class LazyLoaderStereoMIS(nn.Module):
    poses_json: dict
    data_root: str

    def __post_init__(self):
        self.camera = create_camera_model(self.poses_json)
        self.data_root = Path(self.data_root)

    def __getitem__(self, idx):
        frame = self.poses_json["frames"][idx]

        # pose
        pose_blender = None

        # Create rays
        pose_cv = np.array(frame["transform_matrix"])
        rays = self.camera.generate_rays_in_world_coord(pose_cv, fmt="cv2")

        # Load time
        cur_time = torch.tensor(frame["time"])
        cur_time = cur_time.expand(rays.shape[0], 1)
        time_scale = 1.0
        cur_time = time_scale * (cur_time * 2.0 - 1.0)

        # Load RGB
        image_path = self.data_root / f"{frame['file_path']}.png"
        rgb = Image.open(image_path)

        # Load depth
        depth_file_path = frame["depth_file_path"]
        depth = torch.tensor(tiff.imread(depth_file_path))
        depth = depth.view(1, -1).permute(1, 0)  # (h*w, 1) Gray-scale

        return Sample(rgb, depth, pose_blender, rays, cur_time)
