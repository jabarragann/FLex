from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def process_interlaced_frame(frame: np.ndarray):
    height, width, channels = frame.shape

    # Create empty images for left and right frames
    left_image = np.zeros((height // 2, width, channels), dtype=np.uint8)
    right_image = np.zeros((height // 2, width, channels), dtype=np.uint8)

    # Extract even rows for left image, odd rows for right image
    left_image[:, :, :] = frame[0::2, :, :]
    right_image[:, :, :] = frame[1::2, :, :]

    return left_image, right_image


def frame_calculation(start_time: str, end_time: str, video_cv2: VideoOpenCV):
    # Defaults
    if start_time == "00:00:00" and end_time == "00:00:00":
        return 0, video_cv2.total_frames

    start_tuple = tuple(map(int, start_time.split(":")))
    end_tuple = tuple(map(int, end_time.split(":")))

    # Convert time to frame numbers
    # TODO: frames will only be multiples of fps.
    start_frame = (
        int(start_tuple[0] * 3600 + start_tuple[1] * 60 + start_tuple[2])
        * video_cv2.fps
    )
    end_frame = (
        int(end_tuple[0] * 3600 + end_tuple[1] * 60 + end_tuple[2]) * video_cv2.fps
    )

    if start_frame >= video_cv2.total_frames or end_frame >= video_cv2.total_frames:
        print("Error: Start or end time exceeds video duration.")
        return

    return start_frame, end_frame


class VideoOpenCV:
    def __init__(self, video_path: Path):
        self.video_path = video_path
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.cap = cv2.VideoCapture(str(video_path))

        if not self.cap.isOpened():
            raise Exception(f"Error: Could not open video in {video_path}.")

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.print_info()

    def print_info(self):
        print("video information:")
        print(f"loaded video: {self.video_path}")
        print(f"FPS: {self.fps}")
        print(f"Total frames: {self.total_frames}")

    def set_start_frame(self, start_frame):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def is_opened(self):
        return self.cap.isOpened()

    def get_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def __del__(self):
        print("Closing video capture")
        self.cap.release()
