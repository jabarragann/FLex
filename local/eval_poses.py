import torch
import numpy as np
import os
import sys
import json
from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt

from flex.render.util.util import draw_poses
from local.util import eval_poses, sixD_to_mtx, read_poses, end_trajectory_error


def get_cam2world(poses_rot, poses_t, img_idxs=None, initial_pose=None):

    if img_idxs is not None:
        poses_rot = torch.stack([poses_rot[img_idx] for img_idx in img_idxs], dim=0)
        poses_t = torch.stack([poses_t[img_idx] for img_idx in img_idxs], dim=0)
    else:
        poses_rot = torch.stack(list(poses_rot), dim=0)
        poses_t = torch.stack(list(poses_t), dim=0)
    
    return torch.cat([sixD_to_mtx(poses_rot), poses_t[..., None]], dim = -1)


def run_eval_poses(basedir, datadir, file_name, only_end_traj_error=False):

    # load predicted poses
    if "localrf" in basedir:
        with open(os.path.join(basedir, file_name)) as f:
                meta = json.load(f)
        pred_c2w = []
        for frame in meta["frames"]:
            pred_c2w.append(torch.tensor(frame["transform_matrix"]))
        pred_c2w = torch.stack(pred_c2w, 0)[:,:3,:4]

    elif "robust_pose" in basedir:
        poses_arr = read_poses(pose_path=os.path.join(basedir, 'trajectory.freiburg'))
        pred_c2w = torch.tensor(poses_arr)[:,:3,:4] #[1:])[:,:3,:4] # fix duplicate of first entry
        #pred_c2w = None
    else:
        poses_rot = torch.load(os.path.join(basedir, file_name+"_rot.th"),map_location=torch.device('cpu')).detach()
        poses_t = torch.load(os.path.join(basedir, file_name+"_t.th"),map_location=torch.device('cpu')).detach()
        pred_c2w = get_cam2world(poses_rot, poses_t)

    # select test idxs --> for poses use all idxs
    if pred_c2w is None:
        test_idxs = np.arange(0, 1000)
    else:
        test_idxs = np.arange(0, len(pred_c2w))#[::8]

    if only_end_traj_error:
        if not "robust_pose" in basedir:
            with open(os.path.join(datadir, f"transforms_test.json")) as f:
                meta = json.load(f)
                depth_scale = meta["depth_scale_factor"]
            pred_c2w[:,:3,3] *= depth_scale
        rpe_trans, rpe_rot = end_trajectory_error(pred_c2w, savePath=basedir)
        '''
        with open(f"{basedir}/end_pose_error.txt", "w") as f:
            f.write(
                f"RPE-Trans: {rpe_trans}, RPE-Rot: {rpe_rot}\n"
            )
            print(f"RPE-Trans: {rpe_trans}, RPE-Rot: {rpe_rot}")
        '''
        print(f"RPE-Trans: {rpe_trans[-1]}, RPE-Rot: {rpe_rot[-1]}")
        with open(f"{basedir}/end_pose_error_info.txt", "w") as f:
            for i in range(len(rpe_trans)):
                f.write(
                    f"Frame: {i}, RPE-Trans: {rpe_trans[i]}, RPE-Rot: {rpe_rot[i]}\n"
                )
        with open(f"{basedir}/end_pose_error_raw.txt", "w") as f:
            for i in range(len(rpe_trans)):
                f.write(
                    f"{i},{rpe_trans[i]}, {rpe_rot[i]}\n"
                )
                
        # draw trajectory video:
        poses_vis = []
        for pose in tqdm(pred_c2w[::4], desc=f"Drawing Poses ({len(pred_c2w[::4])})"):
            all_poses_pred = torch.cat([pred_c2w, pose[None]], dim=0)
            colours = ["C1"] * pred_c2w.shape[0] + ["C2"]
            # visualize only optimized poses
            all_poses = all_poses_pred
            # make it more visually distinguishable
            pose_vis = draw_poses(all_poses.cpu().float(), colours)
            #pose_vis = cv2.resize(pose_vis, (int(pose_vis.shape[1] * rgb_map.shape[0] / pose_vis.shape[0]), rgb_map.shape[0]))
            poses_vis.append(pose_vis)
        with open(f"{basedir}/posesvideo.mp4", "wb") as f:
            imageio.mimwrite(f, np.stack(poses_vis), fps=10, quality=6, format="mp4", output_params=["-f", "mp4"])
    else:
        ate_rmse, rpe_trans_mean, rpe_rot_mean, trans_error, rpe_trans, rpe_rot = eval_poses(data_dir=datadir, depth_scale=1.0, test_idxs=test_idxs, pred_list=pred_c2w, delta=1, offset=0, savePath=basedir, gen_images=True)

        # save metrics!
        with open(f"{basedir}/pose_mean.txt", "w") as f:
            f.write(
                f"ATE-RMSE: {ate_rmse}, Trans-Error: {trans_error.mean()}, RPE-Trans: {rpe_trans_mean}, RPE-Rot: {rpe_rot_mean}\n"
            )
            print(f"ATE-RMSE: {ate_rmse}, Trans-Error: {trans_error.mean()}, RPE-Trans: {rpe_trans_mean}, RPE-Rot: {rpe_rot_mean}")
            for i in range(len(trans_error)):
                if i<(len(trans_error)-1):
                    f.write(
                        f"Index {i}, Trans-Error: {trans_error[i]}, RPE-Trans: {rpe_trans[i]}, RPE-Rot: {rpe_rot[i]}\n"
                    )
                else:
                    f.write(
                        f"Index {i}, Trans-Error: {trans_error[i]}\n"
                    )






if __name__ == "__main__":
    
    # LocalRF
    # = "Poses_Eval/P2_8_1/localrf_poses"
    #file_name = "transforms_p2_8_1.json"
    # FLex
    file_name = "pose_optim_poses" #"test_poses"
    basedir = "Poses_Eval/24_rev/FLex_poses"
    # Robust Pose
    #basedir = "Poses_Eval/24_rev/robust_pose_poses"
    #file_name = None

    datadir = "Endoscopic_NeRF/data/StereoMIS/24_rev"
    run_eval_poses(basedir, datadir, file_name, only_end_traj_error=True)