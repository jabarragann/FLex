from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
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


@dataclass
class PinHoleCameraParams:
    fx: float
    fy: float
    cx: float
    cy: float
    W: float
    H: float


def point_cloud_to_rgb_no_zbuffer(
    points: np.ndarray, colors: np.ndarray, K: PinHoleCameraParams
):
    # Filter out points behind the camera
    valid = points[:, 2] > 0
    points = points[valid]

    # Projection to image plane
    u = (points[:, 0] * K.fx / points[:, 2]) + K.cx
    v = (points[:, 1] * K.fy / points[:, 2]) + K.cy

    # Round to get pixel coordinates
    u = u.astype(np.int32)
    v = v.astype(np.int32)

    # Optional: filter points that fall outside image bounds
    mask = (u >= 0) & (u < K.W) & (v >= 0) & (v < K.H)
    u = u[mask]
    v = v[mask]
    colors = colors[mask]

    # Initialize image
    rgb_img = np.zeros((K.H, K.W, 3), dtype=np.uint8)

    # Assign color values (overwriting if necessary)
    rgb_img[v, u] = colors

    return rgb_img


def create_point_cloud_with_open3d(
    K: PinHoleCameraParams, rgb, depth, basedir, frame_id: int
):
    # fx = K[0, 0]
    # fy = K[1, 1]
    # cx = K[0, 2]
    # cy = K[1, 2]

    cam_model = o3d.camera.PinholeCameraIntrinsic(
        int(K.W), int(K.H), K.fx, K.fy, K.cx, K.cy
    )

    rgb_im = o3d.geometry.Image(rgb)
    depth_im = o3d.geometry.Image(depth)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_im, depth_im, convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, cam_model)

    o3d.io.write_point_cloud(basedir / f"outputs/frame_{frame_id:04d}_pc.ply", pcd)

    print(cam_model)

    return pcd


def create_point_cloud_with_rays(
    K: PinHoleCameraParams, rgb, depth, basedir, frame_id: int
):
    u, v = np.meshgrid(np.arange(K.W), np.arange(K.H))
    u = u.flatten()
    v = v.flatten()
    depth = depth.flatten()

    # Remove points with zero depth
    valid = depth > 0
    u, v, depth = u[valid], v[valid], depth[valid]

    x = (u - K.cx) * depth / K.fx
    y = (v - K.cy) * depth / K.fy
    z = depth

    points_3d = np.vstack((x, y, z)).T  # Shape: (N, 3)

    colors = rgb[v, u] / 255.0  # Normalize to [0,1]

    # Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(basedir / f"outputs/frame_{frame_id:04d}_pc_rays.ply", pcd)

    return pcd

    ## Optionally it could be implemented with matrix multiplication
    # K = np.array([[fx, 0, cx],
    #             [0, fy, cy],
    #             [0,  0,  1]])
    ## Construct homogeneous 2D pixel coordinates
    # pixels_h = np.vstack((u, v, np.ones_like(u)))
    # # Compute 3D points using inverse projection
    # points_3d = np.linalg.inv(K) @ pixels_h * depth  # Shape: (3, N)
    # points_3d = points_3d.T  # Convert to (N, 3)


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


def compare_projected_pc(K: PinHoleCameraParams, pc1, pc2, basedir, frame_id):
    points = np.asarray(pc1.points)
    colors = (np.asarray(pc1.colors) * 255).astype(np.uint8)  # shape (N, 3)
    img_from_pc = point_cloud_to_rgb_no_zbuffer(points, colors, K)
    # Save or display
    cv2.imwrite(
        str(basedir / f"outputs/projected_pc1_no_zbuffer_{frame_id:04d}.png"),
        cv2.cvtColor(img_from_pc, cv2.COLOR_RGB2BGR),
    )

    points = np.asarray(pc2.points)
    colors = (np.asarray(pc2.colors) * 255).astype(np.uint8)  # shape (N, 3)
    img_from_pc = point_cloud_to_rgb_no_zbuffer(points, colors, K)
    # Save or display
    cv2.imwrite(
        str(basedir / f"outputs/projected_pc2_no_zbuffer_{frame_id:04d}.png"),
        cv2.cvtColor(img_from_pc, cv2.COLOR_RGB2BGR),
    )

    ## PC to RGBD with Open3D
    ## https://www.open3d.org/docs/latest/python_example/geometry/point_cloud/index.html#point-cloud-to-rgbd-py
    ## https://github.com/isl-org/Open3D/issues/2596
    ## Didn't work => project_to_rgbd_image no available.

    # intrinsic = o3d.core.Tensor(
    #     ([[K.fx, 0, K.cx], [0, K.fy, K.cy], [0, 0, 1]])
    # ).to(o3d.core.Dtype.Float32)
    # rgbd_reproj = pc1.project_to_rgbd_image(
    #     intrinsic, K.W, K.H, depth_scale=1.0, depth_trunc=3.0
    # )
    # cv2.imwrite(
    #     str(
    #         basedir
    #         / f"outputs/projected_pc1_no_zbuffer_{rgbd_reproj.color:04d}_o3d.png"
    #     ),
    #     cv2.cvtColor(img_from_pc, cv2.COLOR_RGB2BGR),
    # )


def main():
    basedir = Path(__file__).resolve().parent

    frame_id = 14  # 379 or 14
    rgb_path = f"./sample_data/pc_depth_anything/frame_{frame_id:04d}.png"
    depth_path = f"./sample_data/pc_depth_anything/depth_{frame_id:04d}.png"
    rgb = imageio.imread(basedir / rgb_path)
    disparity = imageio.imread(basedir / depth_path)
    disparity = disparity.astype(np.float32)
    disparity_normalized = (
        disparity / 65535.0
    ) + 1  # Normalize to [1, 2] to avoid div by 0.

    scale = 1000  # Scale to see mesh in Open3D
    depth = scale / disparity_normalized

    W = rgb.shape[1]
    H = rgb.shape[0]
    K = generate_camera_intrinsics(W, H)
    cam_params = PinHoleCameraParams(
        fx=K[0, 0],
        fy=K[1, 1],
        cx=K[0, 2],
        cy=K[1, 2],
        W=W,
        H=H,
    )

    visualize_rgb_and_depth(rgb, depth)
    pc1 = create_point_cloud_with_open3d(cam_params, rgb, depth, basedir, frame_id)
    pc2 = create_point_cloud_with_rays(cam_params, rgb, depth, basedir, frame_id)

    compare_projected_pc(cam_params, pc1, pc2, basedir, frame_id)

    # print(pcd)
    print(rgb.shape)
    print(K)


if __name__ == "__main__":
    main()
