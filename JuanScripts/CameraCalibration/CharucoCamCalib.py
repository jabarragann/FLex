## Camera calibration with a Charuco board with Opencv 4.9.0

from pathlib import Path

import cv2


def show_image(img):
    window_name = "window"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_charuco_corners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

    if len(corners) > 0:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Interpolate Charuco corners
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners, markerIds=ids, image=gray, board=charuco_board
        )

        if retval > 3:  # minimum number of charuco corners
            cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)

        obj_points, img_points = charuco_board.matchImagePoints(
            charuco_corners,
            charuco_ids,
        )
        show_image(frame)
        pass

    ## Calibration options:
    ## Option 1.
    # ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    #     charucoCorners=all_corners,
    #     charucoIds=all_ids,
    #     board=charuco_board,
    #     imageSize=image_size,
    #     cameraMatrix=None,
    #     distCoeffs=None
    # )

    ## Option 2.
    # ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


def detect_chessboard_corners(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if ret:
        cv2.drawChessboardCorners(frame, (9, 6), corners, ret)
        show_image(frame)


script_dir = Path(__file__).parent.resolve()


if __name__ == "__main__":

    # Define Charuco board parameters
    CHARUCO_ROWS = 7
    CHARUCO_COLS = 10
    SQUARE_LENGTH = 0.005  # in meters
    MARKER_LENGTH = 0.003  # in meters

    # Create Charuco board and dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    charuco_board = cv2.aruco.CharucoBoard(
        (CHARUCO_COLS, CHARUCO_ROWS),
        squareLength=SQUARE_LENGTH,
        markerLength=MARKER_LENGTH,
        dictionary=aruco_dict,
    )

    img1 = cv2.imread(
        str(script_dir / "test_images/CharucoNoOcclusions.jpg"), cv2.IMREAD_COLOR
    )
    img2 = cv2.imread(
        str(script_dir / "test_images/CharucoOcclusions.jpg"), cv2.IMREAD_COLOR
    )

    show_image(img1)
    show_image(img2)

    ## Charuco detection worked well with an occluded image.
    detect_charuco_corners(img2)

    ## Chessboard detection with Charuco board is not perfect. He makes some mistakes.
    ## It is also not very fast
    detect_chessboard_corners(img1)
