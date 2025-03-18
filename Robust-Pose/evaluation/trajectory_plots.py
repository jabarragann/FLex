import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("..")
# mpl.use("Qt5Agg")
from core.utils.trajectory import read_freiburg
from core.utils.trajectory_analyzer import TrajectoryAnalyzer
from evaluation.evaluate_ate_freiburg import eval


class Plotter3D:
    def __init__(self):
        self.trajectories = {}

    def add_trajectory(self, method: str, trajectory: np.ndarray):
        x_coord = trajectory[:, 0, 3]
        y_coord = trajectory[:, 1, 3]
        z_coord = trajectory[:, 2, 3]
        formatted_trajectory = dict(x=x_coord, y=y_coord, z=z_coord)
        self.trajectories[method] = formatted_trajectory

    def create_plot(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        for m, t in self.trajectories.items():
            ax.plot(t["x"], t["y"], t["z"], label=m)

        ax.legend()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Trajectories")

    parser.add_argument("base_path", type=str, help="Path to scared dataset.")
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["ours", "ground-truth"],  # "orbslam2",
        help="Folder containing predictions.",
    )
    parser.add_argument(
        "--prealign", action="store_true", help="pre-align trajectories."
    )
    args = parser.parse_args()

    colors = {
        "ground-truth": ["k", 2.5, "dashed"],
        "orbslam2": ["b", 1, "dashdot"],
        "efusion": ["m", 0.5, "solid"],
        "ours": ["goldenrod", 2.5, "solid"],
    }
    d_idx = 1

    keyframe = os.path.basename(args.base_path)
    dataset = os.path.basename(os.path.dirname(args.base_path))
    pose_plotter = TrajectoryAnalyzer(title=dataset + "/" + keyframe)

    freiburg_paths = {
        m: os.path.join(args.base_path, "data", m, "trajectory.freiburg")
        for m in args.methods
    }
    freiburg_paths.update(
        {"ground-truth": os.path.join(args.base_path, "groundtruth.txt")}
    )

    gt_poses = None  # Calculated in eval
    trajectories = {}
    for k, meth in enumerate(freiburg_paths):
        print(meth)
        if meth == "ground-truth":
            pose_arrs = gt_poses.copy()
            if not args.prealign:
                pose_arrs = np.linalg.inv(pose_arrs[0])[None, ...] @ pose_arrs

            trajectories[meth] = pose_arrs
        else:
            assert os.path.isfile(freiburg_paths[meth]), f"{meth} does not exist"
            ate_rmse, rpe_trans, rpe_rot, error, *_, T, gt_poses, _ = eval(
                freiburg_paths["ground-truth"],
                freiburg_paths[meth],
                offset=-4,
                ret_align_T=True,
            )

            print("ATE-RMSE: ", ate_rmse, " mm")
            print("RPE-trans: ", rpe_trans, " mm")
            print("RPE_rot: ", rpe_rot)

            pose_arrs = np.stack(read_freiburg(freiburg_paths[meth]).matrix())
            if args.prealign:
                # align trajectories
                pose_arrs = T[None, ...] @ pose_arrs
            else:
                pose_arrs = np.linalg.inv(pose_arrs[0])[None, ...] @ pose_arrs

            trajectories[meth] = pose_arrs

        n = meth.split("/")[-1]
        pose_plotter.add_pose_trajectory(
            pose_arrs,
            label="ORB-SLAM2" if n == "orbslam2" else n,
            color=colors[n][0],
            linewidth=colors[n][1],
            linestyle=colors[n][2],
        )

    pose_plotter.legend()
    base_path = Path(args.base_path)
    filename = os.path.basename(args.base_path) + ".pdf"
    pose_plotter.write_file(base_path / "data" / filename)
    pose_plotter.show()

    # 3D plot
    plotter3d = Plotter3D()
    for m, t in trajectories.items():
        plotter3d.add_trajectory(m, t)
    plotter3d.create_plot()
