import os
import sys
import cv2
import torch
import numpy as np
import argparse

def gen_empty_masks(dir_path):
    # this function should fill missing masks with the prior one
    folder = sorted(os.listdir(os.path.join(dir_path, 'video_frames')))
    root = os.path.join(dir_path, 'video_frames')
    left_files = [item for item in folder if "l" in item]
    num_files = len(left_files)
    img = cv2.imread(os.path.join(root, left_files[0]))
    masks_folder = os.path.join(dir_path, 'masks')
    os.makedirs(masks_folder)
    for i in left_files:
        mask = np.ones_like(img)[:,:,0]
        cv2.imwrite(os.path.join(masks_folder, (i)), mask)



if __name__ == "__main__":
    
    # Console commands
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str, help='data path') # e.g.: data/StereoMIS/test
    args = parser.parse_args()
    root = os.path.split(os.path.split(os.getcwd())[0])[0]
    args.dir_path = os.path.join(root, args.dir_path)

    gen_empty_masks(args.dir_path)