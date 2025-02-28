import argparse
import os
import sys

import torch
from lietorch import SE3

sys.path.append(os.getcwd())
import glob
import json

import numpy as np
from scipy.spatial.transform import Rotation

from local.eval_poses import get_cam2world


def save_trajectory(trajectory: list, out_dir: str, depth_scale: float):
    # Select the data split.
    i_test = np.arange(len(trajectory))[::8]
    i_train = np.array([i for i in np.arange(int(len(trajectory))) if i not in i_test])

    # convert from OpenGL to OpenCV format
    flip = np.eye(4)
    flip[1, 1] = -1
    flip[2, 2] = -1
    trajectory = np.matmul(trajectory, flip)

    # convert to SE3 representation and save
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "own_poses_train.freiburg"), "w") as f:
        for i in i_train:
            r = Rotation.from_matrix(trajectory[i, :3, :3])
            t = trajectory[i, :3, 3]
            t *= depth_scale  # map from NeRF scale to mm
            t /= 1000  # convert from mm to m
            vec = r.as_quat()
            timestep = i
            f.write(
                f"{timestep} {t[0]} {t[1]} {t[2]} {vec[0]} {vec[1]} {vec[2]} {vec[3]}\n"
            )

    with open(os.path.join(out_dir, "own_poses_test.freiburg"), "w") as f:
        for i in i_test:
            # print(i, trajectory[i])
            # sys.exit()
            r = Rotation.from_matrix(trajectory[i, :3, :3])
            t = trajectory[i, :3, 3]
            t *= depth_scale  # map from NeRF scale to mm
            t /= 1000  # convert from mm to m
            vec = r.as_quat()
            timestep = i
            f.write(
                f"{timestep} {t[0]} {t[1]} {t[2]} {vec[0]} {vec[1]} {vec[2]} {vec[3]}\n"
            )


def convert_flex_poses(input_dir, out_dir, data_dir):
    poses_rot = glob.glob(os.path.join(input_dir, "*poses_rot.th"))
    poses_t = glob.glob(os.path.join(input_dir, "*poses_t.th"))
    poses_rot = torch.load(poses_rot[0], map_location=torch.device("cpu")).detach()
    poses_t = torch.load(poses_t[0], map_location=torch.device("cpu")).detach()
    poses = get_cam2world(poses_rot, poses_t).numpy()

    with open(os.path.join(data_dir, f"transforms_train.json")) as f:
        meta = json.load(f)
        depth_scale = meta["depth_scale_factor"]

    save_trajectory(poses, out_dir, depth_scale)


if __name__ == "__main__":
    print("Start!")
    # console arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Input poses directory")
    parser.add_argument("--out_dir", type=str, help="Save in this directory location")
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Provide data directory on which the poses where trained. Need for proper scaling",
    )
    args = parser.parse_args()
    convert_flex_poses(
        input_dir=args.input_dir, out_dir=args.out_dir, data_dir=args.data_dir
    )
    print("Done!")
