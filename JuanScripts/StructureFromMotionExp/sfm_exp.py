from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_images():
    root_path = "/media/juan95/b0ad3209-9fa7-42e8-a070-b02947a78943/home/camma/JuanData/PigDataset/Pig_dataset/01/video/downsample_motion_1/"
    root_path = Path(root_path)
    right_path = root_path / "left_frames" / "left_0000.png"
    left_path = root_path / "right_frames" / "right_0000.png"

    left_img = cv2.imread(str(left_path))
    right_img = cv2.imread(str(right_path))

    return left_img, right_img


# SIFT features and matching
#############################


def compute_sift_features(img):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors


def match_features(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


# Fundamental matrix calculations
#################################
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


def compute_fundamental_matrix(kp1, kp2, matches):
    if len(matches) < 8:
        raise ValueError("Not enough matches to compute the fundamental matrix")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    # Extract inlier points
    print(f"Inlier values: {mask.sum()}/{mask.size}")
    inlier_pts1 = pts1[mask.ravel() == 1]
    inlier_pts2 = pts2[mask.ravel() == 1]

    fundamental_matrix_error(F, inlier_pts1, inlier_pts2)

    return F, inlier_pts1, inlier_pts2


def stereo_rectify_uncalibrated(left_img, right_img, inliers_pts1, inliers_pts2, F):
    w1, h1 = left_img.shape[1], left_img.shape[0]
    retval, H1, H2 = cv2.stereoRectifyUncalibrated(
        inliers_pts1, inliers_pts2, F, (w1, h1)
    )
    rectified_left = cv2.warpPerspective(left_img, H1, (w1, h1))
    rectified_right = cv2.warpPerspective(right_img, H2, (w1, h1))
    concat = np.concatenate((rectified_left, rectified_right), axis=0)

    window_name = "window"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, concat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Visualize matches
#####################
def draw_epilines(img1, F, pts1, pts2):
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    img1_epilines = img1.copy()

    for r, pt1 in zip(lines1, pts1):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [img1.shape[1], -(r[2] + r[0] * img1.shape[1]) / r[1]])
        img1_epilines = cv2.line(img1_epilines, (x0, y0), (x1, y1), color, 1)
        img1_epilines = cv2.circle(
            img1_epilines, tuple(pt1.astype(np.uint)), 5, color, -1
        )

    img1_epilines = cv2.cvtColor(img1_epilines, cv2.COLOR_BGR2RGB)
    plt.imshow(img1_epilines), plt.show()


def draw_matches(img1, kp1, img2, kp2, matches):
    matched_img = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches[:50],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    plt.figure(figsize=(12, 6))
    plt.imshow(matched_img, cmap="gray")
    plt.axis("off")
    plt.show()


def main():
    left_img, right_img = load_images()

    kp1, desc1 = compute_sift_features(left_img)
    kp2, desc2 = compute_sift_features(right_img)
    matches = match_features(desc1, desc2)

    F, inliers_pts1, inliers_pts2 = compute_fundamental_matrix(kp1, kp2, matches)

    print(F)

    # ## Did not work!
    # stereo_rectify_uncalibrated(left_img, right_img, inliers_pts1, inliers_pts2, F)
    # ## Seems bizarre.
    # draw_epilines(left_img, F, inliers_pts1, inliers_pts2)

    K = np.array(
        [
            [700, 0, left_img.shape[1] // 2],  # Approximate intrinsic matrix
            [0, 700, left_img.shape[0] // 2],
            [0, 0, 1],
        ]
    ).astype(float)
    E = K.T @ F @ K

    retval, R, T, mask = cv2.recoverPose(E, inliers_pts1, inliers_pts2, K)
    print("Rotation Matrix:\n", R)
    print("Translation Vector:\n", T)

    ## Triangulate points and calculate reprojection error
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, T])
    points_4d = cv2.triangulatePoints(P1, P2, inliers_pts1.T, inliers_pts2.T)
    points_3d = points_4d[:3] / points_4d[3]
    points_3d = points_3d.T

    # Reproject points
    reprojected_pts1, _ = cv2.projectPoints(
        points_3d, np.zeros((3, 1)), np.zeros((3, 1)), K, None
    )
    reprojected_pts2, _ = cv2.projectPoints(points_3d, R, T, K, None)

    # Compute reprojection error
    error1 = np.linalg.norm(inliers_pts1 - reprojected_pts1.squeeze(), axis=1)
    error2 = np.linalg.norm(inliers_pts2 - reprojected_pts2.squeeze(), axis=1)
    errors = np.concatenate((error1, error2))

    print(
        f"Mean Reprojection Error: {np.mean(errors):0.04f}\u00b1{np.std(errors):0.04f}"
    )
    print(
        f"max {np.max(errors):0.04f} min {np.min(errors):0.04f} median {np.median(errors):0.04f}"
    )

    # ##Visualize
    # draw_matches(left_img, kp1, right_img, kp2, matches)
    # window_name = "window"
    # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # cv2.imshow(window_name, concat)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
