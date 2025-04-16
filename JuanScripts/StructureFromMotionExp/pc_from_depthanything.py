from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

# PC to RGBD Open3D
# https://www.open3d.org/docs/latest/python_example/geometry/point_cloud/index.html#point-cloud-to-rgbd-py


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


def point_cloud_to_rgb_no_zbuffer(points, colors, W, H, fx, fy, cx, cy):
    # Filter out points behind the camera
    valid = points[:, 2] > 0
    points = points[valid]

    u = (points[:, 0] * fx / points[:, 2]) + cx
    v = (points[:, 1] * fy / points[:, 2]) + cy

    # Round to get pixel coordinates
    u = u.astype(np.int32)
    v = v.astype(np.int32)

    # Optional: filter points that fall outside image bounds
    mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[mask]
    v = v[mask]
    colors = colors[mask]

    # Initialize image
    rgb_img = np.zeros((H, W, 3), dtype=np.uint8)

    # Assign color values (overwriting if necessary)
    rgb_img[v, u] = colors

    return rgb_img


def create_point_cloud_with_open3d(K, rgb, depth, W, H, basedir):
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

    o3d.io.write_point_cloud(basedir / f"outputs/frame_{14:06d}_pc.ply", pcd)

    print(cam_model)

    ## Project back to image
    points = np.asarray(pcd.points)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)  # shape (N, 3)

    img_from_pc = point_cloud_to_rgb_no_zbuffer(points, colors, W, H, fx, fy, cx, cy)

    # Save or display
    cv2.imwrite(
        str(basedir / "outputs/reprojected_rgb_no_zbuffer.png"),
        cv2.cvtColor(img_from_pc, cv2.COLOR_RGB2BGR),
    )


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


def main():
    basedir = Path(__file__).resolve().parent

    # Load images
    rgb = imageio.imread(basedir / "./sample_data/pc_depth_anything/frame_014.png")
    # depth = imageio.imread(basedir/"./sample_data/pc_depth_anything/frame_014_depth.png")
    disparity = np.load(
        "/home/juan95/research/monocular_depth_est/Video-Depth-Anything/outputs_npz/cafiero_03.04_clip4_good_depths.npz"
    )["depths"][13]

    disparity_max = np.max(disparity)
    disparity_min = np.min(disparity)
    disparity_normalized = (
        (disparity - disparity_min) / (disparity_max - disparity_min)
    ) + 1.0  # Normalize to [0, 1]

    depth = 1000 / disparity_normalized
    # depth = disparity

    W = rgb.shape[1]
    H = rgb.shape[0]
    K = generate_camera_intrinsics(W, H)

    visualize_rgb_and_depth(rgb, depth)
    create_point_cloud_with_open3d(K, rgb, depth, W, H, basedir)

    # print(pcd)
    print(rgb.shape)
    print(K)


if __name__ == "__main__":
    main()
