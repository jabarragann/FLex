import sys

sys.path.append("../")
import copy
import os
import time
import warnings

import cv2
import numpy as np
import torch
import wandb
from core.fusion.surfel_map import SurfelMap
from core.pose.pose_estimator import PoseEstimator
from core.RAFT.core.utils.utils import InputPadder
from core.utils.logging import InferenceLogger
from core.utils.trajectory import read_freiburg, save_trajectory
from dataset.dataset_utils import SequentialSubSampler, StereoVideoDataset, get_data
from evaluation.evaluate_ate_freiburg import eval
from lietorch import SE3
from torch.utils.data import DataLoader
from tqdm import tqdm
from viewer.view_renderer import ViewRenderer
from viewer.viewer2d import Viewer2D
from viewer.viewer3d import Viewer3D


def encode_flow(flow, mask):
    flow = 2**15 + flow * (2**8)
    mask &= np.max(flow, axis=-1) < (2**16 - 1)
    mask &= 0 < np.min(flow, axis=-1)
    return np.concatenate(
        [flow.astype(np.uint16), mask[..., None].astype(np.uint16) * (2**16 - 1)],
        axis=-1,
    )


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:, :, 0] += np.arange(w)
    flow_new[:, :, 1] += np.arange(h)[:, np.newaxis]

    res = cv2.remap(
        img, flow_new, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    return res


def compute_fwdbwd_mask(fwd_flow, bwd_flow, alpha_1=0.05, alpha_2=0.5):
    bwd2fwd_flow = warp_flow(bwd_flow, fwd_flow)
    fwd_lr_error = np.linalg.norm(fwd_flow + bwd2fwd_flow, axis=-1)
    fwd_mask = (
        fwd_lr_error
        < alpha_1
        * (np.linalg.norm(fwd_flow, axis=-1) + np.linalg.norm(bwd2fwd_flow, axis=-1))
        + alpha_2
    )

    fwd2bwd_flow = warp_flow(fwd_flow, bwd_flow)
    bwd_lr_error = np.linalg.norm(bwd_flow + fwd2bwd_flow, axis=-1)

    bwd_mask = (
        bwd_lr_error
        < alpha_1
        * (np.linalg.norm(bwd_flow, axis=-1) + np.linalg.norm(fwd2bwd_flow, axis=-1))
        + alpha_2
    )

    return fwd_mask, bwd_mask


def main(args, config):
    device = torch.device("cpu")
    if args.device == "gpu":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            warnings.warn("No GPU available, fallback to CPU")

    if args.log is not None:
        config.update({"keyframe": os.path.split(args.input)[-1]})
        config.update({"dataset": os.path.split(args.input)[-2]})
        wandb.init(project="Alley-OOP", config=config, group=args.log)
        if args.outpath is None:
            args.outpath = wandb.run.dir
    if args.outpath is None:
        """
        try:
            args.outpath = os.path.join(args.input, 'data',f'{config["seq_number"]}', 'infer_trajectory')
        except KeyError:
            args.outpath = os.path.join(args.input, 'data', 'infer_trajectory')
        """
        args.outpath = args.input

    os.makedirs(args.outpath, exist_ok=True)
    os.makedirs(f"{args.outpath}/flow_ds", exist_ok=True)
    os.makedirs(f"{args.outpath}/flow_ds/fwd", exist_ok=True)
    os.makedirs(f"{args.outpath}/flow_ds/bwd", exist_ok=True)
    os.makedirs(f"{args.outpath}/depth_original", exist_ok=True)

    dataset, calib = get_data(
        args.input, config["img_size"], rect_mode=config["rect_mode"]
    )
    # check for ground-truth pose data for logging purposes
    gt_file = os.path.join(args.input, "groundtruth.txt")
    gt_trajectory = read_freiburg(gt_file) if os.path.isfile(gt_file) else None
    init_pose = (
        gt_trajectory[None, args.start]
        if gt_trajectory is not None
        else SE3.Identity(1)
    )

    # adjust for non-8-divisible image resolutions
    adj_img_size = copy.deepcopy(config["img_size"])
    if config["img_size"][0] % 32 != 0 or config["img_size"][1] % 32 != 0:
        os.makedirs(f"{args.outpath}/adj_video_frames", exist_ok=True)
        os.makedirs(f"{args.outpath}/adj_masks", exist_ok=True)
        if adj_img_size[0] % 32 != 0:
            adj_img_size[0] = config["img_size"][0] - (config["img_size"][0] % 32)
        elif adj_img_size[1] % 32 != 0:
            adj_img_size[1] = config["img_size"][1] - (config["img_size"][1] % 32)
    print(f"New Image Size: {adj_img_size}")
    pose_estimator = PoseEstimator(
        config["slam"],
        torch.tensor(calib["intrinsics"]["left"]).to(device),
        baseline=calib["bf"],
        checkpoint=args.checkpoint,
        img_shape=adj_img_size,
        init_pose=init_pose,
    ).to(device)
    if not isinstance(dataset, StereoVideoDataset):
        sampler = SequentialSubSampler(dataset, args.start, args.stop, args.step)
    else:
        warnings.warn(
            "start/stop arguments not supported for video dataset. ignored.",
            UserWarning,
        )
        sampler = None
    loader = DataLoader(
        dataset,
        num_workers=0 if config["slam"]["debug"] else 1,
        pin_memory=True,
        sampler=sampler,
    )

    recorder = InferenceLogger()
    # recorder.set_gt(gt_trajectory)
    with torch.no_grad():
        viewer = None
        if args.viewer == "3d":
            viewer = Viewer3D(
                (2 * config["img_size"][0], 2 * config["img_size"][1]),
                blocking=args.block_viewer,
            )
        elif args.viewer == "2d":
            viewer = Viewer2D(outpath=args.outpath, blocking=args.block_viewer)
        elif args.viewer == "video":
            viewer = ViewRenderer(
                (2 * config["img_size"][1], 2 * config["img_size"][0]),
                outpath=args.outpath,
            )

        trajectory = [{"camera-pose": init_pose, "timestamp": args.start}]
        ##########
        invalid_depth_values, invalid_depth_values_r = 0, 0
        ##########
        if args.metrics_only is None:
            for i, data in enumerate(
                tqdm(
                    loader,
                    total=min(len(dataset), (args.stop - args.start) // args.step),
                )
            ):
                if isinstance(dataset, StereoVideoDataset):
                    limg, rimg, mask, pose_kinematics, img_number = data
                else:
                    limg, rimg, mask, img_number = data
                h, w = limg.shape[2], limg.shape[3]
                if h % 32 != 0 or w % 32 != 0:
                    limg = limg[:, :, : adj_img_size[1], : adj_img_size[0]]
                    rimg = rimg[:, :, : adj_img_size[1], : adj_img_size[0]]
                    mask = mask[:, :, : adj_img_size[1], : adj_img_size[0]]
                    print(limg.shape)
                (
                    pose,
                    scene,
                    flow,
                    weights,
                    depth,
                    flow_bwd,
                    valid_depth,
                    depth_r,
                    valid_depth_r,
                ) = pose_estimator(limg.to(device), rimg.to(device), mask.to(device))

                # save flow and depth maps:
                fbase = f"{str(i).zfill(4)}"
                W, H = config["img_size"][0], config["img_size"][1]
                if flow is not None:
                    flow_fwd = flow[0].permute(1, 2, 0).cpu().numpy()
                    flow_bwd = flow_bwd[0].permute(1, 2, 0).cpu().numpy()

                    mask_fwd, mask_bwd = compute_fwdbwd_mask(flow_fwd, flow_bwd)

                    cv2.imwrite(
                        f"{args.outpath}/flow_ds/fwd/fwd_{fbase}.png",
                        encode_flow(flow_fwd, mask_fwd),
                    )
                    cv2.imwrite(
                        f"{args.outpath}/flow_ds/bwd/bwd_{fbase}.png",
                        encode_flow(flow_bwd, mask_bwd),
                    )
                else:
                    flow_fwd = np.zeros(
                        limg[0, :, :h, :w].permute(1, 2, 0)[..., :2].shape,
                        dtype=np.float32,
                    )
                    flow_bwd = np.zeros(
                        limg[0, :, :h, :w].permute(1, 2, 0)[..., :2].shape,
                        dtype=np.float32,
                    )
                    mask_fwd = np.zeros(
                        limg[0, :, :h, :w].permute(1, 2, 0)[..., 0].shape, dtype=bool
                    )
                    mask_bwd = np.zeros(
                        limg[0, :, :h, :w].permute(1, 2, 0)[..., 0].shape, dtype=bool
                    )

                    cv2.imwrite(
                        f"{args.outpath}/flow_ds/fwd/fwd_{fbase}.png",
                        encode_flow(flow_fwd, mask_fwd),
                    )
                    cv2.imwrite(
                        f"{args.outpath}/flow_ds/bwd/bwd_{fbase}.png",
                        encode_flow(flow_bwd, mask_bwd),
                    )

                # stereo depth for left image
                depth = depth[0].permute(1, 2, 0).cpu().numpy()
                valid_depth = valid_depth[0].permute(1, 2, 0).cpu().numpy()
                # swap value for invalid depth to 0.0
                invalid_depth_values += np.sum(~valid_depth)
                depth[(~valid_depth)] = 0.0
                cv2.imwrite(f"{args.outpath}/depth_original/depth_{fbase}.png", depth)

                print(f"Depth: {np.mean(depth)}")

                if h % 32 != 0 or w % 32 != 0:
                    img_name = f"{i:06d}"
                    cv2.imwrite(
                        f"{args.outpath}/adj_video_frames/{img_name}l.png",
                        cv2.cvtColor(
                            limg[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8),
                            cv2.COLOR_BGR2RGB,
                        ),
                    )
                    cv2.imwrite(
                        f"{args.outpath}/adj_video_frames/{img_name}r.png",
                        cv2.cvtColor(
                            rimg[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8),
                            cv2.COLOR_BGR2RGB,
                        ),
                    )
                    cv2.imwrite(
                        f"{args.outpath}/adj_masks/{img_name}l.png",
                        mask[0].permute(1, 2, 0).cpu().numpy(),
                    )

                # visualization
                if isinstance(viewer, Viewer3D) & (i > 0):
                    curr_pcl = SurfelMap(
                        frame=pose_estimator.get_frame(),
                        kmat=torch.tensor(calib["intrinsics"]["left"]).float(),
                        pmat=pose,
                    ).pcl2open3d(stable=False)
                    curr_pcl.paint_uniform_color([0.5, 0.5, 1.0])
                    canonical_scene = scene.pcl2open3d(stable=False)
                    viewer(pose.cpu(), canonical_scene, add_pcd=curr_pcl)
                elif isinstance(viewer, Viewer2D) & (i > 0):
                    viewer(pose_estimator.get_frame(), weights, flow, i * args.step)
                elif isinstance(viewer, ViewRenderer) & (i > 0):
                    canonical_scene = scene.pcl2open3d(stable=True)
                    viewer(pose.cpu(), canonical_scene)
                trajectory.append({"camera-pose": pose, "timestamp": img_number[0]})

                # logging
                if (args.log is not None) & (i > 0):
                    recorder(scene, pose, step=int(img_number[0]))

            ##################
            print(f"Number of invaldid depth values: {invalid_depth_values}")
            ##################

            save_trajectory(trajectory, args.outpath)
            if scene is not None:
                scene.save_ply(
                    os.path.join(args.outpath, "stable_map.ply"), stable=True
                )
                scene.save_ply(os.path.join(args.outpath, "all_map.ply"), stable=False)
        if args.metrics is not None:
            print("test metrics")
            if os.path.isfile(os.path.join(args.input, "groundtruth.txt")):
                (
                    ate_rmse,
                    rpe_trans,
                    rpe_rot,
                    trans_error,
                    rpe_trans_e,
                    rpe_rot_e,
                ) = eval(
                    os.path.join(args.input, "groundtruth.txt"),
                    os.path.join(args.outpath, "trajectory.freiburg"),
                    offset=0,
                )
                print(f"ATE-RMSE: {ate_rmse}")
                print(f"RPE-Trans: {np.mean(rpe_trans)}")
                print(f"RPE-Rot: {np.mean(rpe_rot)}")
        if args.log is not None:
            wandb.save(os.path.join(args.outpath, "trajectory.freiburg"))
            wandb.save(os.path.join(args.outpath, "map.ply"))
            if os.path.isfile(os.path.join(args.input, "groundtruth.txt")):
                (
                    ate_rmse,
                    rpe_trans,
                    rpe_rot,
                    trans_error,
                    rpe_trans_e,
                    rpe_rot_e,
                ) = eval(
                    os.path.join(args.input, "groundtruth.txt"),
                    os.path.join(args.outpath, "trajectory.freiburg"),
                    offset=-4,
                )
                wandb.define_metric("trans_error", step_metric="frame")
                wandb.define_metric("rpe_trans_e", step_metric="frame")
                wandb.define_metric("rpe_rot_e", step_metric="frame")
                for i, (e1, e2, e3) in enumerate(
                    zip(trans_error, rpe_trans_e, rpe_rot_e)
                ):
                    wandb.log(
                        {
                            "trans_error": e1,
                            "rpe_trans_e": e2,
                            "rpe_rot_e": e3,
                            "frame": i,
                        }
                    )
                wandb.summary["ATE/RMSE"] = ate_rmse
                wandb.summary["RPE/trans"] = rpe_trans
                wandb.summary["RPE/rot"] = rpe_rot

        print("finished")


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="script to run pose estmation")

    parser.add_argument("input", type=str, help="Path to input folder.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../trained/poseNet_2xf8up4b.pth",
        help="Path to trained Pose Estimator Checkpoint.",
    )
    parser.add_argument(
        "--outpath",
        type=str,
        help="Path to output folder. If not provided use input path instead.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../configuration/infer_f2f.yaml",
        help="Configuration file.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="gpu",
        help="select cpu or gpu to run slam.",
    )
    parser.add_argument(
        "--stop", type=int, default=10000000000, help="number of samples to run for."
    )
    parser.add_argument(
        "--start", type=int, default=0, help="at which sample to start slam."
    )
    parser.add_argument("--step", type=int, default=1, help="sub sampling interval.")
    parser.add_argument(
        "--log", default=None, help="wandb group logging name. No logging if none set"
    )
    parser.add_argument(
        "--force_video",
        action="store_true",
        help="force to use video input and recompute depth",
    )
    parser.add_argument(
        "--viewer",
        default="none",
        choices=["none", "2d", "3d", "video"],
        help="select viewer",
    )
    parser.add_argument(
        "--block_viewer", action="store_true", help="block viewer if viewer selected."
    )
    parser.add_argument(
        "--metrics", default=1, help="wandb group logging name. No logging if none set"
    )
    parser.add_argument(
        "--metrics_only",
        default=None,
        help="wandb group logging name. No logging if none set",
    )
    args = parser.parse_args()
    root = os.path.split(os.path.split(os.getcwd())[0])[0]
    args.input = os.path.join(root, args.input)
    with open(args.config) as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    assert os.path.isfile(args.checkpoint), "no valid checkpoint file"

    main(args, config)
