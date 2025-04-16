import cv2
import matplotlib.pyplot as plt
import numpy as np

from flex.render.util.util import decode_flow


def load_flow(path):
    encoded_fwd_flow = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    flow, mask = decode_flow(encoded_fwd_flow)
    return flow, mask


def visualize_optical_flow_arrow(optical_flow, start_frame, scale):
    # Convert flow values back to NumPy array
    flow = np.array(optical_flow)  # Shape: [H, W, 2]

    # Create a copy of the start frame for visualization with flow arrows
    vis_frame_with_arrows = start_frame.copy()

    H, W, _ = flow.shape  # Get flow dimensions

    # Draw optical flow arrows on the start frame
    for y in range(0, H, 50):  # Sample every 100th pixel
        for x in range(0, W, 50):
            dx, dy = flow[y, x]  # Flow vector at (x, y)
            start_point = (x, y)
            end_point = (
                int(x + scale * dx),
                int(y + scale * dy),
            )  # may adjust scale for better visualization
            cv2.arrowedLine(
                vis_frame_with_arrows,
                start_point,
                end_point,
                (0, 255, 0),
                thickness=2,
                tipLength=0.3,
            )

    # Convert BGR to RGB for visualization with Matplotlib
    # vis_frame_with_arrows_rgb = cv2.cvtColor(vis_frame_with_arrows, cv2.COLOR_BGR2RGB)

    return vis_frame_with_arrows


def main():
    frame1_path = "/home/juan95/JuanData/StereoMIS_FLex_juan/P2_8_2_juan_clip/right_frames/right_0000.png"
    frame2_path = "/home/juan95/JuanData/StereoMIS_FLex_juan/P2_8_2_juan_clip/right_frames/right_0001.png"
    bwd_flow_path = "/home/juan95/JuanData/StereoMIS_FLex_juan/P2_8_2_juan_clip/flow_ds_right_frames/bwd/bwd_right_0001.png"
    fwd_flow_path = "/home/juan95/JuanData/StereoMIS_FLex_juan/P2_8_2_juan_clip/flow_ds_right_frames/fwd/fwd_right_0001.png"

    fwd_flow, fwd_mask = load_flow(fwd_flow_path)
    bwd_flow, bwd_mask = load_flow(bwd_flow_path)

    frame1 = cv2.imread(frame1_path)
    frame22 = cv2.imread(frame2_path)

    vis_fwd_flow = visualize_optical_flow_arrow(fwd_flow, frame1, scale=5)
    vis_bwd_flow = visualize_optical_flow_arrow(bwd_flow, frame22, scale=5)

    ## Keypoints selected by hand
    kp1 = (701, 441)
    kp2 = (704, 438)
    cv2.circle(
        frame1, kp1, radius=5, color=(255, 0, 0), thickness=-1
    )  # Blue circle with radius 10
    cv2.circle(
        frame22, kp2, radius=5, color=(255, 0, 0), thickness=-1
    )  # Blue circle with radius 10

    ## Checks
    kp1 = np.array(kp1)
    kp2 = np.array(kp2)
    print(f"Estimated fwd flow: {fwd_flow[kp1[1], kp1[0]]} GT fwd flow: {kp2 - kp1}")
    print(f"Estimated bwd flow: {bwd_flow[kp2[1], kp2[0]]} GT bwd flow: {kp1 - kp2}")

    fig, ax = plt.subplots(2, 2, figsize=(10, 5))

    # Display the first image
    ax[0, 0].imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    ax[0, 0].set_title("Image 1")
    ax[0, 0].axis("off")  # Turn off axis

    # Display the second image
    ax[0, 1].imshow(cv2.cvtColor(frame22, cv2.COLOR_BGR2RGB))
    ax[0, 1].set_title("Image 2")
    ax[0, 1].axis("off")  # Turn off axis

    ax[1, 0].imshow(cv2.cvtColor(vis_fwd_flow, cv2.COLOR_BGR2RGB))
    ax[1, 0].set_title("vis_fwd_flow")
    ax[1, 0].axis("off")  # Turn off axis

    ax[1, 1].imshow(cv2.cvtColor(vis_bwd_flow, cv2.COLOR_BGR2RGB))
    ax[1, 1].set_title("vis_bwd_flow")
    ax[1, 1].axis("off")  # Turn off axis

    # ax[2, 0].imshow(fwd_mask, cmap="gray")
    # ax[2, 0].set_title("fwd_mask")
    # ax[2, 0].axis("off")  # Turn off axis

    # ax[2, 1].imshow(bwd_mask, cmap="gray")
    # ax[2, 1].set_title("bwd_mask")
    # ax[2, 1].axis("off")  # Turn off axis

    plt.show()


if __name__ == "__main__":
    print("OpticalFlowUtils.py")
    main()
