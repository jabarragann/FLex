import sys
sys.path.append('../')
import os
import torch
import numpy as np
from tqdm import tqdm
from dataset.dataset_utils import get_data_images, StereoImageDataset
from torch.utils.data import DataLoader
import cv2


def main(input_path, output_path, step, rect_mode, img_size):

    dataset, calib = get_data_images(input_path, (img_size[0], img_size[1]), sample_video=step, rect_mode=rect_mode)
    assert isinstance(dataset, StereoImageDataset)

    loader = DataLoader(dataset, num_workers=1)

    os.makedirs(os.path.join(output_path, 'video_frames'), exist_ok=True)

    with torch.inference_mode():
        for idx in range(len(dataset)):
            data = dataset[idx]
            limg, rimg, img_number = data

            img_name = f'{idx:06d}'
            cv2.imwrite(os.path.join(output_path, 'video_frames', img_name+'l.png'),
                        limg.squeeze().permute(1,2,0).cpu().numpy().astype(np.uint8))
            cv2.imwrite(os.path.join(output_path, 'video_frames', img_name + 'r.png'),
                        rimg.squeeze().permute(1,2,0).cpu().numpy().astype(np.uint8))
        print('finished')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='script to extract stereo data')

    parser.add_argument(
        'input',
        type=str,
        help='Path to input folder.'
    )
    parser.add_argument(
        '--outpath',
        type=str,
        help='Path to output folder. If not provided use input path instead.'
    )
    parser.add_argument(
        '--rect_mode',
        type=str,
        choices=['conventional', 'pseudo'],
        default='conventional',
        help='rectification mode, use pseudo for SCARED'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=[640, 512],
        help='desired image dimensions',
        nargs='+',
    )
    args = parser.parse_args()
    ## hack file folder
    print(os.getcwd())
    print(os.path.split(os.path.split(os.path.split(os.path.split(os.getcwd())[0])[0])[0])[0])
    #root = os.path.split(os.path.split(os.path.split(os.path.split(os.getcwd())[0])[0])[0])[0]
    root = os.path.split(os.path.split(os.getcwd())[0])[0]
    basedir = os.path.join(root, args.input)
    args.input = basedir
    if args.outpath is None:
        args.outpath = args.input
    ####################
    
    datasets = np.genfromtxt(os.path.join(args.input, 'sequences.txt'), skip_header=1, delimiter=',', dtype=str)
    datasets = datasets[None, ...] if datasets.shape == (2,) else datasets
    for d in datasets:
        print(f'extract {d[0]}')
        try:
            main(os.path.join(args.input, d[0]), os.path.join(args.outpath, d[0]), 1, args.rect_mode, img_size=args.img_size)
        except IndexError:
            pass
        except AssertionError:
            print(f"skip {d[0]}, already extracted")
