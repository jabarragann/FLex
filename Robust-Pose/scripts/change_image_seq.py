import cv2
import numpy as np
import os
import sys


def change_image_seq(input_folder):
    # doubles the sequence length by reversing the sequence
    file_paths = sorted(os.listdir(os.path.join(input_folder, 'video_frames')))
    # use only left images 
    left_files = [item for item in file_paths if "l" in item]
    right_files = [item for item in file_paths if "r" in item]
    # select which frames to keep
    left_files = left_files[::2]
    right_files = right_files[::2]
    if len(left_files)!=0:
        current_root = os.path.join(input_folder, 'video_frames')
        new_root = os.path.join(input_folder, 'video_frames_new')
        os.makedirs(new_root)
        for idx in range(len(left_files)):
            left_img = cv2.imread(os.path.join(current_root, left_files[idx]))
            right_img = cv2.imread(os.path.join(current_root, right_files[idx]))
            cv2.imwrite(os.path.join(new_root, left_files[idx]),
                        left_img.astype(np.uint8))
            cv2.imwrite(os.path.join(new_root, right_files[idx]),
                        right_img.astype(np.uint8))

if __name__ == '__main__':

    input_folder = "data/P3_3_huge"

    change_image_seq(input_folder)