from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Tuple

import imageio
import numpy as np
from PIL import Image

sys.path.append(os.getcwd())
import argparse
from pathlib import Path

from torchvision import transforms as T
from tqdm.auto import tqdm

from local.util import read_poses


def generate_test_video(file_paths, basedir):
    transform = T.ToTensor()
    all_images = []
    for file_path in file_paths:
        img = Image.open(file_path)

        img = transform(img)  # (3, h, w)
        img = img.permute(1, 2, 0)  # (h*w, 3) RGB
        img = (img * 255).numpy().astype(np.uint8)

        all_images += [img]

    tqdm.write("DEMO|Generate rendering gif...")
    imageio.mimwrite(
        basedir / "test_video.mp4",
        np.stack(np.array(all_images[::2])),
        fps=20,
        format="FFMPEG",
        quality=8,
    )


def create_transforms(
    i_train,
    poses,
    basedir,
    with_tool_mask,
    with_flow_data,
    rgb_paths,
    depth_paths,
    tool_mask_paths,
    fwd_paths,
    bwd_paths,
    all_frames,
    i_test,
    dataset_meta: DatasetMetadata,
    which_cam="right",
    split="train",
):
    dataset_dict = {}

    # Camera parameters
    dataset_dict["w"] = dataset_meta.camera_model.w
    dataset_dict["h"] = dataset_meta.camera_model.h
    dataset_dict["camera_model"] = dataset_meta.camera_model.camera_model
    dataset_dict["fl_x"] = dataset_meta.camera_model.fl_x
    dataset_dict["fl_y"] = dataset_meta.camera_model.fl_y
    dataset_dict["cx"] = dataset_meta.camera_model.cx
    dataset_dict["cy"] = dataset_meta.camera_model.cy
    dataset_dict["depth_scale_factor"] = dataset_meta.depth_scale_factor

    frames = []
    for i in i_train:
        if i > len(depth_paths) - 1:
            break
        item = {}
        item["file_path"] = str(rgb_paths[i])
        item["transform_matrix"] = poses[i, :, :4].tolist()
        item["transform_matrix"].append([0, 0, 0, 1])
        item["time"] = float(i) / (len(rgb_paths) - 1)  # correct version

        item["depth_file_path"] = str(depth_paths[i])

        if with_tool_mask:
            item["tool_mask_path"] = str(tool_mask_paths[i])

        if with_flow_data:
            if i < (len(all_frames) - 1):
                item["fwd_file_path"] = str(fwd_paths[i + 1])
            else:
                item["fwd_file_path"] = str(fwd_paths[0])
            item["bwd_file_path"] = str(bwd_paths[i])

            item["fwd_mask"] = 1 if i < (len(all_frames) - 1) else 0
            item["bwd_mask"] = 1 if i != 0 else 0

            item["prev_test"] = 1 if (i - 1 in i_test) else 0
            item["post_test"] = 1 if (i + 1 in i_test) else 0

        frames.append(item)

    dataset_dict["frames"] = frames

    file_name = basedir / ("transforms_" + split + ".json")
    with open(file_name, "w") as outfile:
        json.dump(dataset_dict, outfile, indent=4)


@dataclass
class DatasetMetadata:
    camera_model: CameraModel
    depth_scale_factor: float
    center_shift: tuple[float]


@dataclass
class CameraModel:
    w: int
    h: int
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    camera_model: str


def create_hexd(
    scene="23",
    factor=1,
    flow_data=False,
    data_type="stereomis_mono",
    which_cam="right",
    basedir="data/StereoMIS_FLex_juan/",
    stereomis_gt=False,
    tool_mask=True,
    own_poses=False,
    gen_gif=False,
    calib_img_size=None,
):

    basedir = Path(basedir)
    basedir = basedir / str(scene)

    # load poses
    poses_arr = read_poses(pose_path=basedir / "groundtruth.txt")
    length = len(poses_arr)
    poses = poses_arr

    # Select the data split.
    i_test = np.arange(length)[::8]
    i_train = np.array([i for i in np.arange(int(length)) if i not in i_test])
    all_frames = np.arange(length)

    # Load frames
    rgb_base_path = basedir / f"{which_cam}_frames"
    rgb_paths = sorted(rgb_base_path.glob("*.png"))
    if len(rgb_paths) == 0:
        raise RuntimeError(f"No images found in the folder {rgb_base_path}")
    if gen_gif:
        generate_test_video(rgb_paths, basedir)

    if flow_data:
        flow_base_path = basedir / f"{which_cam}_frames_flow_ds"
        fwd_paths = sorted((flow_base_path / "fwd").glob("*.png"))
        bwd_paths = sorted((flow_base_path / "bwd").glob("*.png"))
    if tool_mask:
        tool_mask_paths = sorted(os.listdir(os.path.join(basedir, "masks")))

    # Set camera intrinsics
    H, W, _ = np.array(Image.open(rgb_paths[0])).shape
    camera_model = CameraModel(
        w=W,
        h=H,
        fl_x=600,
        fl_y=600,
        cx=W // 2,
        cy=H // 2,
        camera_model="OPENCV",
    )

    ## Preprocess poses
    # Convert from OpenCV to OpenGL Format
    print(f"Poses Trans. Max. Before: {np.max(poses[:, :, 3])}")
    print(f"Poses Trans. Min. Before: {np.min(poses[:, :, 3])}")
    flip = np.eye(4)
    flip[1, 1] = -1
    flip[2, 2] = -1
    poses = np.matmul(poses, flip)

    # Scale and recenter poses
    scale = 1.0
    poses, scale, center_shift = center_poses_0(poses)

    print(f"ReScale: {scale}")
    print(f"ReCenter: {center_shift}")
    print(f"Poses Shape: {poses.shape}")

    dataset_meta = DatasetMetadata(
        camera_model=camera_model,
        depth_scale_factor=scale.item(),
        center_shift=center_shift.tolist(),
    )

    depth_base_path = basedir / f"{which_cam}_depth/depth_frames"
    depth_file_paths = sorted(depth_base_path.glob("*.png"))
    tool_mask_paths = depth_file_paths

    # create dict for each frame:
    create_transforms(
        i_train,
        poses,
        basedir,
        tool_mask,
        flow_data,
        rgb_paths,
        depth_file_paths,
        tool_mask_paths,
        fwd_paths,
        bwd_paths,
        all_frames,
        i_test,
        split="train",
        dataset_meta=dataset_meta,
    )
    create_transforms(
        i_test,
        poses,
        basedir,
        tool_mask,
        flow_data,
        rgb_paths,
        depth_file_paths,
        tool_mask_paths,
        fwd_paths,
        bwd_paths,
        all_frames,
        i_test,
        split="test",
        dataset_meta=dataset_meta,
    )
    create_transforms(
        all_frames,
        poses,
        basedir,
        tool_mask,
        flow_data,
        rgb_paths,
        depth_file_paths,
        tool_mask_paths,
        fwd_paths,
        bwd_paths,
        all_frames,
        i_test,
        split="all",
        dataset_meta=dataset_meta,
    )


def center_poses_0(poses):
    center = poses[..., 3].mean(0)
    poses[..., 3] = poses[..., 3] - center
    max_poses = abs(poses[..., 3]).max(0)
    scale = max_poses.max(0)
    print(f"Max poses along translation axis: {max_poses} and scale value: {scale}")
    poses[..., 3] = (poses[..., 3]) / scale

    return poses, scale, center


if __name__ == "__main__":
    print("Start!")
    # console arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, help="Scene name")
    parser.add_argument("--data_type", type=str, help="Dataset name")
    parser.add_argument(
        "--factor", type=int, default=1, help="Size ratio to initial images"
    )
    parser.add_argument("--flow_data", action="store_true")
    parser.add_argument("--tool_mask", action="store_true")
    parser.add_argument("--stereomis_gt", action="store_true")
    parser.add_argument("--own_poses", action="store_true")
    parser.add_argument("--generate_gif", action="store_true")
    parser.add_argument(
        "--calib_img_size",
        nargs="+",
        type=int,
        default=None,
        help="set image dimensions, epsecially needed when changing image via cropping etc. to fix calibration file",
    )
    args = parser.parse_args()
    create_hexd(
        scene=args.scene,
        factor=args.factor,
        flow_data=args.flow_data,
        data_type=args.data_type,
        stereomis_gt=args.stereomis_gt,
        tool_mask=args.tool_mask,
        own_poses=args.own_poses,
        gen_gif=args.generate_gif,
        calib_img_size=args.calib_img_size,
    )
    print("Done!")

    # Example
    # python flex/dataloader/preprocess_data.py --scene 'P2_8_1' --data_type 'stereomis' --flow_data --tool_mask
