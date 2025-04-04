import cv2
import numpy as np

np.set_printoptions(precision=4, suppress=True)

if __name__ == "__main__":
    W = 640
    H = 480

    K = np.array(
        [
            [70, 0, W // 2],  # Approximate intrinsic matrix
            [0, 75, H // 2],
            [0, 0, 1],
        ]
    ).astype(float)

    identity = np.identity(3)
    rvec, _ = cv2.Rodrigues(identity)
    tvec = np.zeros((3, 1))

    # print(rvec)

    p_3d = np.array([[15.0, -5.0, 35.0]])

    p_img1, _ = cv2.projectPoints(p_3d, rvec, tvec, K, None)

    p_img2 = K @ p_3d.T
    p_img2 = p_img2 / p_img2[2]

    print("Projection to image plane")
    print(p_img1)
    print(p_img2)

    K_inv = np.linalg.inv(K)
    print("K")
    print(K)
    print("inverse K")
    print(K_inv)

    # Reproject back to 3D world
    direction_vec = K_inv @ p_img2
    depth = p_3d[0, 2]
    p_3d_2 = direction_vec * depth

    print("direction vector")
    print(direction_vec)

    print("Back projection to 3D world")
    print(p_3d_2)
