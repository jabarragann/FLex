import glob
import os
import warnings
from typing import Tuple

import cv2
import numpy as np
import torch
from dataset.transforms import ResizeStereo
from torch.utils.data import Dataset


def mask_specularities(img, mask=None, spec_thr=0.96):
    spec_mask = img.sum(axis=-1) < (3 * 255 * spec_thr)
    mask = mask & spec_mask if mask is not None else spec_mask
    mask = cv2.erode(mask.astype(np.uint8), kernel=np.ones((11, 11)))
    return mask


class StereoDataset(Dataset):
    def __init__(self, input_folder: str, img_size: Tuple):
        super().__init__()
        self.imgs = sorted(
            glob.glob(os.path.join(input_folder, "video_frames*", "*l.png"))
        )

        assert len(self.imgs) > 0
        self.transform = ResizeStereo(img_size)

    def __getitem__(self, item):
        img_l = cv2.cvtColor(cv2.imread(self.imgs[item]), cv2.COLOR_BGR2RGB)
        img_r_path = (
            self.imgs[item].replace("l.png", "r.png").replace("video_frames", "right")
        )
        img_r = cv2.cvtColor(cv2.imread(img_r_path), cv2.COLOR_BGR2RGB)
        img_number = os.path.basename(self.imgs[item]).split("l.png")[0]
        mask = cv2.resize(
            cv2.imread(
                self.imgs[item].replace("video_frames", "masks"), cv2.IMREAD_GRAYSCALE
            ),
            dsize=(img_l.shape[1], img_l.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        mask = mask > 0
        # mask specularities
        mask = mask_specularities(img_l, mask)
        # to torch tensor
        img_l = torch.tensor(img_l).permute(2, 0, 1).float()
        img_r = torch.tensor(img_r).permute(2, 0, 1).float()
        mask = torch.tensor(mask).unsqueeze(0)

        data = self.transform(img_l, img_r, mask)

        return (*data, img_number)

    def __len__(self):
        return len(self.imgs)
