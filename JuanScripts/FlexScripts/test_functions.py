from pathlib import Path

import torch

from flex.render.render import get_cam2world
from flex.render.util.util import get_all_poses
from local.util import absolute_trajectory_error


def plot_poses():
    pass


def juan_pose_evaluations(
    train_dataset, test_dataset, poses_t, poses_rot, save_path: str
):
    Path(save_path).mkdir(exist_ok=True)

    ## Load GT poses
    total_frames = len(test_dataset) + len(train_dataset.poses)
    gt_poses, train_img_idxs = get_all_poses(train_dataset, test_dataset, total_frames)

    ## Process predicted poses
    cam2world = get_cam2world(poses_rot.detach().cpu(), poses_t.detach().cpu())
    pred_poses = cam2world
    row_to_add = torch.tensor([0, 0, 0, 1], dtype=pred_poses.dtype).unsqueeze(0)
    pred_poses = torch.cat(
        [pred_poses, row_to_add.expand(pred_poses.size(0), -1, -1)], dim=1
    )

    gt_poses = gt_poses.detach().cpu().numpy()
    pred_poses = pred_poses.detach().cpu().numpy()

    gt_poses[:, :3, 3] = gt_poses[:, :3, 3] * 10
    pred_poses[:, :3, 3] = pred_poses[:, :3, 3] * 10

    ate_rmse, trans_error, transform, valid = absolute_trajectory_error(
        gt_poses.copy(),
        pred_poses.copy(),
        ret_align_T=True,
        ignore_failed_pos=True,
        preprocessed=False,
        savePath=save_path,
    )

    exit()
