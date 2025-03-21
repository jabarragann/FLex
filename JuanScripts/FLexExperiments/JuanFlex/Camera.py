from dataclasses import dataclass

import numpy as np
import torch

from flex.dataloader.ray_utils import get_ray_directions_blender, get_rays


@dataclass
class CameraModel:
    w: int
    h: int
    cx: float
    cy: float
    focal: float

    def __post_init__(self):
        self._calculate_rays_in_cam_coord()

    def _calculate_rays_in_cam_coord(self):
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

    def generate_rays_in_world_coord(self, c2w: np.ndarray, fmt: str = "cv2"):
        if fmt != "cv2":
            raise NotImplementedError("Only opencv format is supported")

        blender2opencv = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

        pose_blender = c2w @ blender2opencv
        pose_blender = torch.FloatTensor(pose_blender)

        # Create rays
        rays_o, rays_d = get_rays(
            self.directions, pose_blender
        )  # Get rays, both (h*w, 3).
        rays = torch.cat([rays_o, rays_d], 1)

        return rays


def create_camera_model(poses_json: dict, downsample: float = 1.0) -> CameraModel:
    # fix img_wh dim:
    w, h = poses_json["w"], poses_json["h"]
    w, h = int(w), int(h)
    cx = poses_json["cx"] / downsample
    cy = poses_json["cy"] / downsample
    focal = poses_json["fl_x"] / downsample
    camera_model = CameraModel(w, h, cx, cy, focal)

    return camera_model
