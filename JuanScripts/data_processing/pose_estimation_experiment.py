# from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from matplotlib import pyplot as plt
from natsort import natsorted
from PoseUtils import read_freiburg


def visualize_two_rgb(img1, img2):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax = ax.reshape(1, 2)

    ax[0, 0].imshow(img1)
    ax[0, 0].set_title("Image 0")
    ax[0, 0].axis("off")  # Turn off axis

    # Display the second image
    ax[0, 1].imshow(img2)
    ax[0, 1].set_title("Image 40")
    ax[0, 1].axis("off")  # Turn off axis

    plt.show()


def visualize_rgb_and_depth(rgb, depth):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax = ax.reshape(1, 2)

    ax[0, 0].imshow(rgb)
    ax[0, 0].set_title("Image 1 with Circle")
    ax[0, 0].axis("off")  # Turn off axis

    # Display the second image
    ax[0, 1].imshow(depth, cmap="gray")
    ax[0, 1].set_title("Image 2")
    ax[0, 1].axis("off")  # Turn off axis

    plt.show()


def draw_key_points(img: np.ndarray, kp: np.ndarray, color=(255, 0, 0)):
    for p in range(kp.shape[0]):
        cv2.circle(img, tuple(kp[p].astype(int)), radius=5, color=color, thickness=-1)


@dataclass
class PinHoleCameraParams:
    fx: float
    fy: float
    cx: float
    cy: float
    W: float
    H: float


@dataclass
class Dataset:
    base_path: Path
    debug: bool = True
    debug_max_frames: int = 50

    def __post_init__(self):
        # Load poses.
        self.c2w, stamps = read_freiburg(
            path, ret_stamps=True, to_SE3=False, relative=True
        )
        # plot_3d_trajectory(self.c2w[:700], stamps)

        # Load RGB
        self.rgb_frames = self.load_images(mode="rgb")
        self.depth_frames = self.load_images(mode="depth")

    def load_depth(self, path: str) -> np.ndarray:
        disparity = imageio.imread(path)
        disparity = disparity.astype(np.float32)

        # Image encoded in a 16-bit PNG format
        disparity_normalized = (
            disparity / 65535.0
        ) + 1  # Normalize to [1, 2] to avoid div by 0.
        scale = 1000  # Scale to see mesh in Open3D
        depth = scale / disparity_normalized

        return depth

    def load_images(self, mode: str):
        """
        Mode can be either 'rgb' , 'depth' or 'flow'
        """
        frames = []

        if mode == "rgb":
            frames_path = self.base_path / "right_frames"
            loading_function = imageio.imread
        elif mode == "depth":  # Load depth
            frames_path = self.base_path / "right_depth/depth_frames"
            loading_function = self.load_depth
        elif mode == "flow":
            pass
        else:
            raise ValueError("Mode should be either 'rgb' or 'depth'")

        frames_list = natsorted(list(frames_path.glob("*.png")))

        print(f"loading {mode} frames")
        for frame_path in frames_list:
            current_frame = loading_function(frame_path)
            frames.append(current_frame)

            if self.debug and len(frames) > self.debug_max_frames:
                break

        frames = np.stack(frames, axis=0)

        return frames


def backproject_to_3d(K: PinHoleCameraParams, kp: np.ndarray, depth: np.ndarray):
    """
    Backproject 2D keypoints to 3D points using depth information.
    """
    # Convert keypoints to homogeneous coordinates
    kp_homogeneous = np.hstack((kp, np.ones((kp.shape[0], 1))))

    # Compute the inverse of the camera intrinsic matrix
    K_inv = np.linalg.inv(np.array([[K.fx, 0, K.cx], [0, K.fy, K.cy], [0, 0, 1]]))

    # Backproject to 3D points
    points_3d = (K_inv @ kp_homogeneous.T).T * depth[kp[:, 1], kp[:, 0]].reshape(-1, 1)

    return points_3d


def project_to_2d(K: PinHoleCameraParams, points_3d: np.ndarray):
    """
    Project 3D points to 2D image coordinates using the camera intrinsic matrix.
    """

    # Compute the camera intrinsic matrix
    K_matrix = np.array([[K.fx, 0, K.cx], [0, K.fy, K.cy], [0, 0, 1]])

    # Project to 2D image coordinates
    points_2d_homogeneous = (K_matrix @ points_3d.T).T

    # Normalize to get pixel coordinates
    points_2d = (
        points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2][:, np.newaxis]
    )

    return points_2d


def register_point_clouds(A, B):
    """
    Estimate rotation R and translation t such that: B ≈ R @ A + t
    A, B: Nx3 arrays of corresponding 3D points
    Returns: R, t
    """

    assert A.shape == B.shape

    # Compute centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - centroid_A
    BB = B - centroid_B
    # Compute covariance matrix
    H = AA.T @ BB
    # SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    # Translation
    t = centroid_B - R @ centroid_A
    return R, t


def manually_seleted_keypoints():
    # Point = namedtuple("Point", ["x", "y"])

    # Keypoints for frame 0
    # Keypoints are in (x,y) coordinates
    kp1 = [[818, 640], [610, 950], [643, 974], [553, 680], [569, 717], [675, 818], [731, 786], [699, 440],
           [293, 569], [458, 512]]  # fmt: skip
    # Keypoints for frame 40
    kp2 = [[918, 363], [654, 720], [687, 747], [606, 431], [619, 473], [729, 587], [795, 554], [802, 118],
           [326, 264], [515, 207]]  # fmt: skip

    kp1 = np.array(kp1).astype(float)
    kp2 = np.array(kp2).astype(float)
    return kp1, kp2


def fundamental_matrix_error(F, inlier_pts1, inlier_pts2):
    ones = np.ones((inlier_pts1.shape[0], 1))
    pts1_hom = np.hstack([inlier_pts1, ones])  # Convert to homogeneous coordinates
    pts2_hom = np.hstack([inlier_pts2, ones])

    errors = np.abs(np.sum(pts2_hom.T * (F @ pts1_hom.T), axis=0))
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    max_error = np.max(errors)

    print("Fundamental matrix error report:")
    print(f"Error: {mean_error: 0.5f}\u00b1{std_error: 0.5f}")
    print(f"Max error: {max_error: 0.5f} min error: {np.min(errors): 0.5f}")

    return mean_error, std_error


##### Experiment 3
###################
def find_registration_from_2_frames(dataset: Dataset):
    kp1, kp2 = manually_seleted_keypoints()

    frame_id = 0
    rgb1 = dataset.rgb_frames[frame_id]
    depth1 = dataset.depth_frames[frame_id]

    frame_id = 40
    rgb2 = dataset.rgb_frames[frame_id]
    depth2 = dataset.depth_frames[frame_id]
    pose = dataset.c2w[frame_id]

    W, H = rgb1.shape[1], rgb1.shape[0]
    K = PinHoleCameraParams(
        fx=999,
        fy=999,
        cx=W / 2,
        cy=H / 2,
        W=W,
        H=H,
    )

    ### Experiment 2
    ##############################
    draw_key_points(rgb1, kp1, color=(0, 255, 0))
    draw_key_points(rgb2, kp2, color=(0, 255, 0))
    kp1_3d = backproject_to_3d(K, kp1, depth1)
    kp2_3d = backproject_to_3d(K, kp2, depth2)

    print(kp2_3d[0])

    R, t = register_point_clouds(kp1_3d, kp2_3d)
    print(kp2_3d.shape)

    est_p = R @ kp1_3d.T + t.reshape(-1, 1)
    est_p = est_p.T

    est_p_2d = project_to_2d(K, est_p)

    print("transformed points")
    print(est_p)
    print("kp2")
    print(kp2_3d)
    print(est_p_2d.shape)

    reproj_error = np.linalg.norm(est_p_2d - kp2, axis=1)
    print(f"reprojection error {np.mean(reproj_error)} ± {np.std(reproj_error)}")

    draw_key_points(rgb2, est_p_2d, color=(0, 0, 255))
    visualize_two_rgb(rgb1, rgb2)

    ## Difficult to compared gt poses with the estimated one.
    print(R)
    print(t)
    print(np.linalg.inv(pose))


##### Experiment 3
###################
def find_registration_from_2_frames_with_fundamental_mat(dataset: Dataset):
    kp1, kp2 = manually_seleted_keypoints()

    frame_id = 0
    rgb1 = dataset.rgb_frames[frame_id]
    # depth1 = dataset.depth_frames[frame_id]

    frame_id = 40
    rgb2 = dataset.rgb_frames[frame_id]
    # depth2 = dataset.depth_frames[frame_id]
    # pose = dataset.c2w[frame_id]

    W, H = rgb1.shape[1], rgb1.shape[0]
    K_inst = PinHoleCameraParams(
        fx=999,
        fy=999,
        cx=W / 2,
        cy=H / 2,
        W=W,
        H=H,
    )

    K = np.array(
        [[K_inst.fx, 0, K_inst.cx], [0, K_inst.fy, K_inst.cy], [0, 0, 1]]
    ).astype(float)

    ### Experiment 3 - traditional sfm pipeline
    ##############################

    # Compute the fundamental matrix
    F, mask = cv2.findFundamentalMat(kp1, kp2, method=cv2.FM_8POINT)
    fundamental_matrix_error(F, kp1, kp2)

    E = K.T @ F @ K

    # R,T transforms from the first camera's coordinate system to the second camera's coordinate system.
    retval, R, T, mask = cv2.recoverPose(E, kp1, kp2, K)

    print("Rotation Matrix:\n", R)
    print("Translation Vector:\n", T)

    ## Triangulate points
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])  # Camera 0 is the world frame
    P2 = K @ np.hstack([R, T])
    points_4d = cv2.triangulatePoints(P1, P2, kp1.T, kp2.T)
    points_3d_in_frame0 = points_4d[:3] / points_4d[3]
    points_3d_in_frame0 = points_3d_in_frame0.T

    # Project points to 2d
    est_kp1, _ = cv2.projectPoints(
        points_3d_in_frame0, np.zeros((3, 1)), np.zeros((3, 1)), K, None
    )
    est_kp2, _ = cv2.projectPoints(points_3d_in_frame0, R, T, K, None)

    # Reprojection error
    points_3d_in_frame40 = R @ points_3d_in_frame0.T + T.reshape(-1, 1)
    points_3d_in_frame40 = points_3d_in_frame40.T
    est_kp2_2 = project_to_2d(K_inst, points_3d_in_frame40)
    reproj_error = np.linalg.norm(est_kp2_2 - kp2, axis=1)
    print(f"reprojection error {np.mean(reproj_error)} ± {np.std(reproj_error)}")

    assert np.all(np.isclose(est_kp2.squeeze(), est_kp2_2))

    ## manually selected keypoints
    draw_key_points(rgb1, kp1, color=(0, 255, 0))
    draw_key_points(rgb2, kp2, color=(0, 255, 0))
    # Reprojected keypoints
    draw_key_points(rgb1, est_kp1.squeeze(), color=(0, 255, 200))
    draw_key_points(rgb2, est_kp2.squeeze(), color=(0, 200, 255))
    visualize_two_rgb(rgb1, rgb2)

    ## Traditional sfm pipeline
    # Returns points in 3d from matched keypoints and the relative camera pose.
    return points_3d_in_frame0, R, T


def use_gt_to_project_pc_from_frame2_frame(dataset: Dataset):
    kp1, kp2 = manually_seleted_keypoints()

    frame_id = 0
    rgb1 = dataset.rgb_frames[frame_id]
    # depth1 = dataset.depth_frames[frame_id]

    frame_id = 40
    rgb2 = dataset.rgb_frames[frame_id]
    depth2 = dataset.depth_frames[frame_id]
    pose = dataset.c2w[frame_id]

    W, H = rgb1.shape[1], rgb1.shape[0]
    K = PinHoleCameraParams(
        fx=999,
        fy=999,
        cx=W / 2,
        cy=H / 2,
        W=W,
        H=H,
    )

    # Experiment 1
    #############################

    depth = depth2[kp2[0, 1], kp2[0, 0]]

    # Project keypoints to 3D
    x = (kp2[0, 0] - K.cx) * depth / K.fx
    y = (kp2[0, 1] - K.cy) * depth / K.fy
    z = depth

    points_3d = np.vstack((x, y, z, 1.0))  # Shape: (N, 3)
    # pose[:3, 3] = pose[:3, 3] / 1000

    print(f"Keypoint 1: {kp1}, Keypoint 2: {kp2}")
    print(f"K matrix: fx: {K.fx}, fy: {K.fy}, cx: {K.cx}, cy: {K.cy}")
    print(f"kp2 3d: {points_3d.T}")
    print(pose)

    p_converted = pose @ points_3d
    print(f"p converted: {p_converted.T}")

    # Project converted points to 2d
    u = (p_converted[0] * K.fx / p_converted[2]) + K.cx
    v = (p_converted[1] * K.fy / p_converted[2]) + K.cy
    print(f"u, v: {u}, {v}")

    cv2.circle(
        rgb1, kp1[0], radius=5, color=(255, 255, 0), thickness=-1
    )  # Blue circle with radius 10

    cv2.circle(
        rgb1, (int(u[0]), int(v[0])), radius=5, color=(255, 255, 255), thickness=-1
    )  # Blue circle with radius 10

    cv2.circle(
        rgb2, kp2[0], radius=5, color=(255, 255, 0), thickness=-1
    )  # Blue circle with radius 10

    # visualize_rgb_and_depth(rgb1, depth1)
    visualize_two_rgb(rgb1, rgb2)


if __name__ == "__main__":
    path = "/home/juan95/JuanData/StereoMIS/P2_8/groundtruth.txt"
    img_folder = Path("/home/juan95/JuanData/StereoMIS_FLex_juan/P2_8_2_juan_clip")

    dataset = Dataset(img_folder)

    ### Experiment 3 - with fundamental matrix
    find_registration_from_2_frames_with_fundamental_mat(dataset)

    ### Experiment 2 - with depth
    ###############################
    # find_registration_from_2_frames(dataset)

    ### Experiment 1 - didn't work
    # Poses scaled might not correspond with the scale of the monodepth.
    ##############################
    # use_gt_to_project_pc_from_frame2_frame(dataset)
