from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import open3d as o3d


# Approximate intrinsic matrix
def generate_camera_intrinsics(w, h, f=999):
    K = np.array(
        [
            [f, 0, w // 2],
            [0, f, h // 2],
            [0, 0, 1],
        ]
    ).astype(float)

    return K


if __name__ == "__main__":
    basedir = Path(__file__).resolve().parent

    # Load images
    rgb = imageio.imread(basedir / "./sample_data/pc_depth_anything/frame_014.png")
    # depth = imageio.imread(basedir/"./sample_data/pc_depth_anything/frame_014_depth.png")
    depth = np.load(
        "/home/juan95/research/monocular_depth_est/Video-Depth-Anything/outputs_npz/cafiero_03.04_clip4_good_depths.npz"
    )["depths"][13]

    W = rgb.shape[1]
    H = rgb.shape[0]
    K = generate_camera_intrinsics(W, H)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    cam_model = o3d.camera.PinholeCameraIntrinsic(int(W), int(H), fx, fy, cx, cy)

    rgb_im = o3d.geometry.Image(rgb)
    depth_im = o3d.geometry.Image(depth)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_im, depth_im, convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, cam_model)

    o3d.io.write_point_cloud(basedir / f"frame_{14:06d}_pc.ply", pcd)

    # print(pcd)
    print(rgb.shape)
    print(cam_model)
    print(K)
