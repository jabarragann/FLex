import numpy as np
import torch
from lietorch import SE3


def plot_camera_orientation(ax, pose, scale=0.1):
    """Plot a single camera pose with coordinate axes."""
    origin = pose[:3, 3]
    x_axis = pose[:3, 0] * scale
    y_axis = pose[:3, 1] * scale
    z_axis = pose[:3, 2] * scale

    ax.quiver(*origin, *x_axis, color="r", length=scale)
    ax.quiver(*origin, *y_axis, color="g", length=scale)
    ax.quiver(*origin, *z_axis, color="b", length=scale)


def plot_3d_trajectory(poses, time_stamps):
    import matplotlib.pyplot as plt

    # from mpl_toolkits.mplot3d import Axes3D
    # Sample 3D trajectory data
    # t = time_stamps
    x = poses[:, 0, 3]
    y = poses[:, 1, 3]
    z = poses[:, 2, 3]  # Z-coordinates (height)

    # Create 3D figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the trajectory
    ax.plot(x, y, z, label="3D Trajectory", color="b")

    # plot orientations
    for i in range(0, len(poses), 20):
        plot_camera_orientation(ax, poses[i], scale=2.0)

    # Add labels
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("3D Trajectory Plot")

    # Show legend
    ax.legend()

    # Show plot
    plt.show()


def read_freiburg(
    path: str, ret_stamps=False, no_stamp=False, to_SE3=False, relative=False
):
    """Read poses

    Parameters
    ----------
    path : str
    ret_stamps : bool, optional
    no_stamp : bool, optional
    to_SE3 : bool, optional
       Convert to lietorch tensor else numpy, by default False
    relative : bool, optional
        Express with respect to first camera frame., by default True

    Returns
    -------
    _type_
        _description_
    """
    with open(path) as f:
        data = f.read()
        lines = data.replace(",", " ").replace("\t", " ").split("\n")
        list = [
            [v.strip() for v in line.split(" ") if v.strip() != ""]
            for line in lines
            if len(line) > 0 and line[0] != "#"
        ]

    if no_stamp:
        trans_slice = slice(0, 3, 1)
        rot_slice = slice(3)
    else:
        trans_slice = slice(1, 4, 1)
        rot_slice = slice(4, 8)  # [4:8]

        time_stamp = [i[0] for i in list if len(i) > 0]
        try:
            time_stamp = (
                np.asarray([int(i.split(".")[0] + i.split(".")[1]) for i in time_stamp])
                * 100
            )
        except IndexError:
            time_stamp = np.asarray([int(i) for i in time_stamp])

    trans = np.asarray([i[trans_slice] for i in list if len(i) > 0], dtype=float)
    trans *= 1000.0  # m to mm
    quat = np.asarray([i[rot_slice] for i in list if len(i) > 0], dtype=float)

    if to_SE3:
        trans = torch.from_numpy(trans)
        quat = torch.from_numpy(quat)
        pose_se3 = SE3.InitFromVec(torch.cat((trans, quat), dim=-1))

        if relative:
            raise (NotImplementedError)

    else:  # Keep as numpy
        from scipy.spatial.transform import Rotation as R

        rot = R.from_quat(quat)
        pose_se3 = np.zeros((quat.shape[0], 4, 4))
        pose_se3[:, :3, :3] = rot.as_matrix()
        pose_se3[:, :3, 3] = trans
        pose_se3[:, 3, 3] = 1.0

        if relative:
            T_1w = np.linalg.inv(pose_se3[0])

            # Didn't work the einsum
            # pose_se3_v1 = np.einsum("ijk,ik->ij", T_1w, pose_se3)
            pose_se3 = T_1w @ pose_se3
            # assert np.isclose(T_1w @ pose_se3[5], pose_se3_v2[5]).all()

    if ret_stamps:
        return pose_se3, time_stamp
    else:
        return pose_se3


def test_scipy_np_poses_equivalence(path):

    pose_se3, stamps = read_freiburg(path, ret_stamps=True, to_SE3=True)

    pose_scipy, stamps = read_freiburg(path, ret_stamps=True, to_SE3=False)

    all = []
    for i in range(20):
        are_close = np.allclose(pose_se3[i].matrix().numpy(), pose_scipy[i])
        all.append(are_close)
        if not are_close:
            print(f"Pose {i} is not close!")
            print(pose_se3[i].matrix().numpy())
            print(pose_scipy[i])

    all = np.array(all)

    if np.all(all):
        print("All values correct")
    assert np.all(all)


np.set_printoptions(precision=4, suppress=True)

if __name__ == "__main__":
    # Example usage
    path = "/home/juan95/JuanData/StereoMIS/P2_8/groundtruth.txt"

    test_scipy_np_poses_equivalence(path)

    pose_normalized, stamps = read_freiburg(
        path, ret_stamps=True, to_SE3=False, relative=True
    )
    plot_3d_trajectory(pose_normalized[:700], stamps)

    print("First pose")
    print(pose_normalized[0])
    print("Second pose")
    print(pose_normalized[1])

    T_12 = pose_normalized[1]
    T_21 = np.linalg.inv(T_12)
    T_13 = pose_normalized[2]

    T_23 = T_21 @ T_13
    T_13_calc = T_12 @ T_23

    print(np.isclose(T_13, T_13_calc))
