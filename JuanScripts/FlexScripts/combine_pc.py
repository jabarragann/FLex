from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from pathlib import Path
from typing import List

import open3d as o3d
from imageio.v2 import imread, imwrite
from test_functions import load_predicted_poses

from flex.dataloader.stereomis_dataset import MonoDepthLoader
from JuanScripts.StructureFromMotionExp.pc_from_depthanything import (
    PinHoleCameraParams,
    point_cloud_to_rgb_no_zbuffer,
    rays_to_pc,
)


def camera_matrix():
    w = 640
    h = 512

    fov = 85.6
    fov = fov * math.pi / 180
    init_focal = w / math.tan(fov / 2) / 2
    focal_offset = 1.0
    center_rel = 0.5
    cx = w * center_rel
    cy = h * center_rel
    focal = init_focal * focal_offset

    print(f"Focal: {focal}, CX: {cx}, CY: {cy}")

    K_fmt1 = np.array(
        [
            [focal, 0, cx],
            [0, focal, cy],
            [0, 0, 1],
        ]
    ).astype(float)

    K_fmt2 = PinHoleCameraParams(
        fx=K_fmt1[0, 0],
        fy=K_fmt1[1, 1],
        cx=K_fmt1[0, 2],
        cy=K_fmt1[1, 2],
        W=w,
        H=h,
    )

    return K_fmt1, K_fmt2


def save_point_cloud(points_3d, colors, save_path: Path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(save_path, pcd)


def load_rgb(list_of_paths: list[Path]):
    print("loading rgb")
    frames = []
    for path in list_of_paths:
        f = imread(path)
        frames.append(f)

    return frames


def load_depth(list_of_paths: list[Path], json_path: Path):
    print("loading depth")
    with open(json_path) as f:
        min_max = json.load(f)

    depth_loader = MonoDepthLoader(
        min_disparity=min_max["min"], max_disparity=min_max["max"]
    )
    frames = []
    for path in list_of_paths:
        f = depth_loader.load_depth(path)
        frames.append(f)
    return frames


def load_poses(poses_path: list[Path]):
    model_name = "FLexPO_P2_8_juan"

    poses_rot = torch.load(poses_path / f"{model_name}_poses_rot.th")
    poses_t = torch.load(poses_path / f"{model_name}_poses_t.th")

    pred_poses = load_predicted_poses(poses_t, poses_rot)

    pred_poses = pred_poses.detach().cpu().numpy()

    return pred_poses


def gen_and_save_framei_pc(
    rgb,
    depth,
    K: PinHoleCameraParams,
    idx: int,
    output_dir: Path,
    uniform_color: int = 0,
):
    points, colors = rays_to_pc(rgb, depth, K)
    colors_uniform = np.zeros_like(colors)
    colors_uniform[:, uniform_color] += 1.0
    pc_path = output_dir / f"frame_{idx:04d}_pc_rays.ply"
    save_point_cloud(points, colors, pc_path)

    return points, colors, colors_uniform


def combine_pc_main():
    base_path = Path(
        "/media/juan95/b0ad3209-9fa7-42e8-a070-b02947a78943/home/camma/JuanData/StereoMIS_FLex_juan/P2_8_2_juan_clip_FLex"
    )
    rgb_path = base_path / "right_frames"
    depth_path = base_path / "right_depth" / "depth_frames"
    poses_path = Path(
        "/home/juan95/research/neural_rendering/FLex/logs/FLexPO/p2_8_2/FLexPO_P2_8_juan-20250501-141539"
    )
    json_depth_path = depth_path / "depth_min_max.json"

    rgb_path = sorted(rgb_path.glob("*.png"))
    depth_path = sorted(depth_path.glob("*.png"))

    rgb = load_rgb(rgb_path[:50])
    depth = load_depth(depth_path[:50], json_depth_path)
    poses = load_poses(poses_path)

    _, K = camera_matrix()

    print(poses.shape)

    output_path = Path(__file__).parent / "outputs"
    output_path.mkdir(exist_ok=True)

    ### Convert poses from blender to opencv format
    flip = np.eye(4)
    flip[1, 1] = -1
    flip[2, 2] = -1
    poses_opencv = np.matmul(poses, flip)

    ### Move poses to the first frame coordinate frames
    T_0_world = np.linalg.inv(poses_opencv[0])
    poses_opencv = T_0_world @ poses_opencv

    ### Generate point clouds from two frames
    points_0, colors_0, colors_0_uniform = gen_and_save_framei_pc(
        rgb[0], depth[0], K, 0, output_path, uniform_color=0
    )
    gen_rgb_0 = point_cloud_to_rgb_no_zbuffer(
        points_0, (colors_0 * 255).astype(np.uint8), K
    )
    imwrite(
        output_path / f"pc_rgb_{0:04d}.png",
        gen_rgb_0,
    )

    idx = 40
    points_30, colors_30, colors_30_uniform = gen_and_save_framei_pc(
        rgb[idx], depth[idx], K, idx, output_path, uniform_color=1
    )
    gen_rgb_30 = point_cloud_to_rgb_no_zbuffer(
        points_30, (colors_30 * 255).astype(np.uint8), K
    )
    imwrite(
        output_path / f"pc_rgb_{idx:04d}.png",
        gen_rgb_30,
    )

    ### Warp point cloud from one frame to another
    points_30_hom = np.concatenate(
        (points_30, np.ones((points_30.shape[0], 1))), axis=1
    )
    new_point_30 = poses_opencv[idx] @ points_30_hom.T
    new_point_30 = new_point_30.T[:, :3]

    ### Saved projected warped point cloud
    gen_rgb_30_warped = point_cloud_to_rgb_no_zbuffer(
        new_point_30, (colors_30 * 255).astype(np.uint8), K
    )
    imwrite(
        Path(__file__).parent / f"outputs/pc_rgb_{idx:04d}_warped.png",
        gen_rgb_30_warped,
    )

    gen_rgb_30_warped = cv2.cvtColor(gen_rgb_30_warped, cv2.COLOR_RGB2BGR)
    blended_img = cv2.addWeighted(gen_rgb_0, 0.5, gen_rgb_30_warped, 0.5, 0)
    imwrite(
        Path(__file__).parent / f"outputs/pc_rgb_{0:04d}_blended.png",
        blended_img,
    )

    ### Save combined point clouds
    pc_path = output_path / "combined_pc_with_pose_aligment.ply"
    concat_points = np.concatenate((points_0, new_point_30), axis=0)
    concat_colors = np.concatenate((colors_0, colors_30), axis=0)
    save_point_cloud(concat_points, concat_colors, pc_path)

    pc_path = output_path / "combined_pc_with_no_aligment.ply"
    concat_points = np.concatenate((points_0, points_30), axis=0)
    concat_colors = np.concatenate((colors_0, colors_30), axis=0)
    save_point_cloud(concat_points, concat_colors, pc_path)


if __name__ == "__main__":
    combine_pc_main()
