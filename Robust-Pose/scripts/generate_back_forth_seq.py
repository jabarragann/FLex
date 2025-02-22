import cv2
import numpy as np
import os
import sys


def generate_back_forth_seq(input_folder):
    # doubles the sequence length by reversing the sequence
    file_paths = sorted(os.listdir(os.path.join(input_folder, 'video_frames')))
    # use only left images 
    left_files = [item for item in file_paths if "l" in item]
    right_files = [item for item in file_paths if "r" in item]
    count = len(left_files)
    if len(left_files)!=0:
        current_root = os.path.join(input_folder, 'video_frames')
        for idx in range(len(left_files)-1, -1, -1):
            left_img = cv2.imread(os.path.join(current_root, left_files[idx]))
            right_img = cv2.imread(os.path.join(current_root, right_files[idx]))
            img_name = f'{count:06d}'
            cv2.imwrite(os.path.join(current_root, img_name+'l.png'),
                        left_img.astype(np.uint8))
            cv2.imwrite(os.path.join(current_root, img_name+'r.png'),
                        right_img.astype(np.uint8))
            count += 1


if __name__ == '__main__':

    input_folder = "data/miti/24_rev"

    generate_back_forth_seq(input_folder)
            