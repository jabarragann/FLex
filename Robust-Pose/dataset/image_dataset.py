import cv2
import os
import sys
import json
from torch.utils.data import IterableDataset
from dataset.transforms import ResizeStereo, Compose
from typing import Tuple, Callable
import numpy as np
import torch
from core.utils.trajectory import read_freiburg
from dataset.stereo_dataset import mask_specularities
from lietorch import SE3


class StereoImageDataset(IterableDataset):
    def __init__(self, limage_folder:str, rimage_folder:str, img_size:Tuple=None, rectify: Callable=None,):
        super().__init__()
        self.limage_folder = limage_folder
        self.rimage_folder = rimage_folder
        assert os.path.exists(self.limage_folder)
        assert os.path.exists(self.rimage_folder)
        self.rectify = rectify
        self.transform = ResizeStereo(img_size)
        self.limage_files = sorted(os.listdir(os.path.join(limage_folder)))
        self.rimage_files = sorted(os.listdir(os.path.join(rimage_folder)))
        assert len(self.limage_files) == len(self.rimage_files)
        self.length = len(self.limage_files)


        self._parse_images()

    def __getitem__(self, idx):

        return self.all_img_left[idx], self.all_img_right[idx], self.all_masks[idx] 

    def _parse_images(self):
        self.all_img_left, self.all_img_right, self.all_masks = [], [], []
        for i in range(len(self.limage_files)):
            img_left = cv2.imread(os.path.join(self.limage_folder, self.limage_files[i]))
            img_right = cv2.imread(os.path.join(self.rimage_folder, self.rimage_files[i]))

            mask = torch.tensor(mask_specularities(img_left)).unsqueeze(0)
            img_left = torch.tensor(img_left).permute(2, 0, 1).float()
            img_right = torch.tensor(img_right).permute(2, 0, 1).float()

            if self.transform is not None:
                img_left, img_right, mask = self.transform(img_left, img_right, mask)

            if self.rectify is not None:
                img_left, img_right = self.rectify(img_left, img_right)

            self.all_img_left.append(img_left)
            self.all_img_right.append(img_right)
            self.all_masks.append(mask)


    def __len__(self):
        return self.length
