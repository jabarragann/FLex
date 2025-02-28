import copy
import json
import os
import sys

import cv2
import numpy as np
import tifffile as tiff
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

from flex.render.util.util import decode_flow

from .ray_utils import get_ray_directions_blender, get_rays, ndc_rays_blender, read_pfm

blender2opencv = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def trans_t(t):
    return torch.Tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]
    ).float()


def rot_phi(phi):
    return torch.Tensor(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ]
    ).float()


def rot_theta(th):
    return torch.Tensor(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ]
    ).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        torch.Tensor(
            np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        )
        @ c2w
        @ blender2opencv
    )
    return c2w


class MitiDataset(Dataset):
    def __init__(
        self,
        datadir,
        split="train",
        downsample=2.0,
        is_stack=False,
        cal_fine_bbox=False,
        N_vis=-1,
        time_scale=1.0,
        scene_bbox_min=[-1.0, -1.0, -1.0],
        scene_bbox_max=[1.0, 1.0, 1.0],
        N_random_pose=1000,
        cfg=None,
    ):
        self.root_dir = datadir
        self.split = split
        self.downsample = downsample
        # self.img_wh = (int(800 / downsample), int(800 / downsample))
        self.is_stack = is_stack
        self.N_vis = N_vis  # evaluate images for every N_vis images

        self.time_scale = time_scale
        self.world_bound_scale = 1.1

        self.near = cfg.data.near  # 0.01#2.0 # adjust
        self.far = cfg.data.far  # 1.0#6.0 # adjust
        self.near_far = [self.near, self.far]
        self.cfg = cfg

        self.define_transforms()  # transform to torch.Tensor

        self.scene_bbox = torch.tensor([scene_bbox_min, scene_bbox_max])
        self.blender2opencv = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

        self.depth_data = cfg.data.depth_data
        self.bpixel_mask_data = cfg.data.bpixel_mask
        self.pixel_feat_data = cfg.model.use_pixel_feat_sup
        self.load_flow = self.cfg.data.flow_data

        self.white_bg = True
        self.ndc_ray = cfg.data.use_ndc  # False
        if self.ndc_ray:
            self.white_bg = False

        self.read_meta()  # Read meta data

        # Calculate a more fine bbox based on near and far values of each ray.
        if cal_fine_bbox:
            xyz_min, xyz_max = self.compute_bbox()
            self.scene_bbox = torch.stack((xyz_min, xyz_max), dim=0)

        self.define_proj_mat()

        self.N_random_pose = N_random_pose
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        # Generate N_random_pose random poses, which we could render depths from these poses and apply depth smooth loss to the rendered depth.
        if split == "train":
            self.init_random_pose()

    def init_random_pose(self):
        # Randomly sample N_random_pose radius, phi, theta and times.
        radius = np.random.randn(self.N_random_pose) * 0.1 + 4
        phi = np.random.rand(self.N_random_pose) * 360 - 180
        theta = np.random.rand(self.N_random_pose) * 360 - 180
        random_times = self.time_scale * (torch.rand(self.N_random_pose) * 2.0 - 1.0)
        self.random_times = random_times

        # Generate rays from random radius, phi, theta and times.
        self.random_rays = []
        for i in range(self.N_random_pose):
            random_poses = pose_spherical(theta[i], phi[i], radius[i])
            rays_o, rays_d = get_rays(self.directions, random_poses)
            self.random_rays += [torch.cat([rays_o, rays_d], 1)]

        self.random_rays = torch.stack(self.random_rays, 0).reshape(
            -1, *self.img_wh[::-1], 6
        )

    def compute_bbox(self):
        print("compute_bbox_by_cam_frustrm: start")
        xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
        xyz_max = -xyz_min
        if (
            self.split == "test"
            or self.cfg.data.datasampler_type == "hierach"
            or self.is_stack
        ):
            rays_o = self.all_rays.reshape(
                self.all_rays.shape[0] * self.all_rays.shape[1], self.all_rays.shape[2]
            )[:, 0:3]
            viewdirs = self.all_rays.reshape(
                self.all_rays.shape[0] * self.all_rays.shape[1], self.all_rays.shape[2]
            )[:, 3:6]
        else:
            rays_o = self.all_rays[:, 0:3]
            viewdirs = self.all_rays[:, 3:6]
        pts_nf = torch.stack(
            [rays_o + viewdirs * self.near, rays_o + viewdirs * self.far]
        )
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0, 1)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0, 1)))
        print("compute_bbox_by_cam_frustrm: xyz_min", xyz_min)
        print("compute_bbox_by_cam_frustrm: xyz_max", xyz_max)
        print("compute_bbox_by_cam_frustrm: finish")
        xyz_shift = (xyz_max - xyz_min) * (self.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
        return xyz_min, xyz_max

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth

    def read_meta(self):
        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json")) as f:
            self.meta = json.load(f)

        """
        self.focal = (
            0.5 * 800 / np.tan(0.5 * self.meta["camera_angle_x"])
        )  # original focal length
        self.focal *= (
            self.img_wh[0] / 800
        )  # modify focal length to match size self.img_wh
        """
        if self.cfg.model.localrf:
            lower_idx = max(
                round(
                    len(self.meta["frames"])
                    * ((0.03) * self.cfg.model.model_num - 0.015)
                ),
                0,
            )
            upper_idx = lower_idx + round(len(self.meta["frames"]) * (0.03))
            self.meta["frames"] = self.meta["frames"][lower_idx:upper_idx]

        if self.cfg.debug_mode:
            if self.split == "train":
                self.meta["frames"] = self.meta["frames"][
                    :100
                ]  # [:28]#[0:53]#[:10] # only debug
            else:
                self.meta["frames"] = self.meta["frames"][
                    :15
                ]  # [:8]#[:4]#[0:8]#[:2] # only debug
            print("Debug Mode activated!")

        w, h = self.meta["w"], self.meta["h"]
        # fix img_wh dim:
        self.img_wh = (int(w), int(h))
        self.cx = self.meta["cx"] / self.downsample
        self.cy = self.meta["cy"] / self.downsample
        self.focal = self.meta["fl_x"] / self.downsample
        self.depth_scale_factor = self.meta["depth_scale_factor"]

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions_blender(
            h,
            w,
            [self.focal, self.focal],
            [
                self.cx,
                self.cy,
            ],  # test cx and cy instead of standard center TODO: remove if experiment failed
        )  # (h, w, 3)
        if not self.ndc_ray:
            self.directions = self.directions / torch.norm(
                self.directions, dim=-1, keepdim=True
            )
        """
        self.intrinsics = torch.tensor(
            [[self.focal, 0, w / 2], [0, self.focal, h / 2], [0, 0, 1]]
        ).float()
        """
        self.intrinsics = torch.tensor(
            [[self.focal, 0, self.cx], [0, self.focal, self.cy], [0, 0, 1]]
        )

        self.image_paths = []
        self.depth_paths = []
        self.poses = []
        self.all_rays = []
        self.all_times = []
        self.all_rgbs = []
        self.all_depths = []
        self.all_pixel_feat = []
        self.all_fwd = []
        self.all_bwd = []
        self.all_fwd_mask = []
        self.all_bwd_mask = []
        self.prev_test = []
        self.post_test = []
        self.all_loss_weights = []

        img_eval_interval = (
            1 if self.N_vis < 0 else len(self.meta["frames"]) // self.N_vis
        )
        idxs = list(range(0, len(self.meta["frames"]), img_eval_interval))
        for i in tqdm(
            idxs, desc=f"Loading data {self.split} ({len(idxs)})"
        ):  # img_list:#
            frame = self.meta["frames"][i]
            pose = np.array(frame["transform_matrix"]) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            self.image_paths += [image_path]
            img = Image.open(image_path)

            img = self.transform(img)  # (3, h, w)
            img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB

            self.all_rgbs += [img]

            # handling of depth files:
            if self.depth_data:
                depth_file_path = frame["depth_file_path"]
                depth = torch.tensor(tiff.imread(depth_file_path))
                depth = depth.view(1, -1).permute(1, 0)  # (h*w, 1) Gray-scale
                self.all_depths += [depth]

            if self.pixel_feat_data:
                # pixel features
                pixel_feat_path = frame["pixel_desc_path"]
                pixel_feat = np.load(pixel_feat_path)["arr_0"][
                    : self.cfg.model.pixel_feat_dim, :
                ]
                pixel_feat_tensor = torch.tensor(pixel_feat)
                del pixel_feat  # save memory
                pixel_feat_tensor = pixel_feat_tensor.view(
                    self.cfg.model.pixel_feat_dim, -1
                ).permute(
                    1, 0
                )  # (h*w, self.cfg.pixel_feat_dim)
                self.all_pixel_feat += [pixel_feat_tensor]

                # pixel feature mask
                pixel_feat_mask_path = os.path.join(
                    os.path.split(self.root_dir)[0], "descriptor-silk-mask.npz"
                )
                pixel_feat_mask_np = np.load(pixel_feat_mask_path)["arr_0"]
                self.pixel_feat_mask = torch.tensor(pixel_feat_mask_np)
                del pixel_feat_mask_np  # save memory
                self.pixel_feat_mask = self.pixel_feat_mask.view(1, -1).permute(
                    1, 0
                )  # (h*w, 1)
                self.pixel_feat_mask = self.pixel_feat_mask.repeat(
                    len(self.image_paths), 1
                )  # (num.images*h*w,1)

            if self.load_flow:
                # optical flow forwards and backwards
                fwd_file_path = frame["fwd_file_path"]
                bwd_file_path = frame["bwd_file_path"]

                encoded_fwd_flow = cv2.imread(fwd_file_path, cv2.IMREAD_UNCHANGED)
                encoded_bwd_flow = cv2.imread(bwd_file_path, cv2.IMREAD_UNCHANGED)
                flow_scale = h / encoded_fwd_flow.shape[0]

                encoded_fwd_flow = cv2.resize(
                    encoded_fwd_flow, tuple((w, h)), interpolation=cv2.INTER_AREA
                )
                encoded_bwd_flow = cv2.resize(
                    encoded_bwd_flow, tuple((w, h)), interpolation=cv2.INTER_AREA
                )

                fwd_flow, fwd_mask = decode_flow(encoded_fwd_flow)
                bwd_flow, bwd_mask = decode_flow(encoded_bwd_flow)

                fwd_flow = fwd_flow * flow_scale
                bwd_flow = bwd_flow * flow_scale

                fwd_flow = self.transform(fwd_flow).view(2, -1).permute(1, 0)
                bwd_flow = self.transform(bwd_flow).view(2, -1).permute(1, 0)
                fwd_mask = (
                    self.transform(fwd_mask).view(1, -1).permute(1, 0)
                )  # .reshape(-1, 1)
                bwd_mask = (
                    self.transform(bwd_mask).view(1, -1).permute(1, 0)
                )  # .reshape(-1, 1)
                fwd_mask2 = torch.full_like(fwd_mask, frame["fwd_mask"])
                bwd_mask2 = torch.full_like(bwd_mask, frame["bwd_mask"])
                fwd_mask = torch.logical_and(fwd_mask, fwd_mask2)
                bwd_mask = torch.logical_and(bwd_mask, bwd_mask2)
                self.prev_test.append(torch.tensor([frame["prev_test"]]))
                self.post_test.append(torch.tensor([frame["post_test"]]))

            else:
                fwd_flow, fwd_mask, bwd_flow, bwd_mask = None, None, None, None

            self.all_fwd.append(fwd_flow)
            self.all_bwd.append(bwd_flow)
            self.all_fwd_mask.append(fwd_mask)
            self.all_bwd_mask.append(bwd_mask)

            rays_o, rays_d = get_rays(self.directions, c2w)  # Get rays, both (h*w, 3).
            if self.ndc_ray:
                rays_o, rays_d = ndc_rays_blender(h, w, self.focal, 1.0, rays_o, rays_d)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
            cur_time = torch.tensor(
                frame["time"]
                if "time" in frame
                else float(i) / (len(self.meta["frames"]) - 1)
            ).expand(rays_o.shape[0], 1)
            self.all_times += [cur_time]

        if self.bpixel_mask_data:
            bpixel_mask_path = os.path.join(
                os.path.split(self.root_dir)[0], "black_pixel_mask/mask.png"
            )
            self.bpixel_mask = self.transform(Image.open(bpixel_mask_path))
            # set mask to binary by filtering out all pixels below 255
            self.bpixel_mask = self.bpixel_mask >= 255 / 255
            self.bpixel_mask = self.bpixel_mask.view(1, -1).permute(1, 0)  # (h*w, 1)
            self.bpixel_mask = self.bpixel_mask.repeat(
                len(self.image_paths), 1
            )  # (num.images*h*w,1)

        # Test loss weights
        all_laplacian = []
        for img in self.all_rgbs:
            img2 = img.permute(1, 0).view(3, h, w).permute(1, 2, 0)
            all_laplacian.append(
                np.ones_like(img2[..., 0])
                * cv2.Laplacian(
                    cv2.cvtColor(
                        ((img2.numpy()) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
                    ),
                    cv2.CV_32F,
                ).var()
            )
            all_laplacian[-1] = all_laplacian[-1].reshape(-1, 1)
        self.all_loss_weights = [torch.tensor(laplacian) for laplacian in all_laplacian]

        self.poses = torch.stack(self.poses)
        if self.split == "train" and self.load_flow:
            self.prev_test = torch.stack(self.prev_test)
            self.post_test = torch.stack(self.post_test)
        #  self.is_stack stacks all images into a big chunk, with shape (N, H, W, 3).
        #  Otherwise, all images are kept as a set of rays with shape (N_s, 3), where N_s = H * W * N
        if not self.is_stack:
            self.all_rays = torch.cat(
                self.all_rays, 0
            )  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(
                self.all_rgbs, 0
            )  # (len(self.meta['frames])*h*w, 3)
            self.all_times = torch.cat(self.all_times, 0)
            if self.depth_data:
                self.all_depths = torch.cat(self.all_depths, 0)
            if self.pixel_feat_data:
                self.all_pixel_feat = torch.cat(
                    self.all_pixel_feat, 0
                )  # (len(self.meta['frames])*h*w, self.cfg.model.pixel_feat_dim)
            if self.load_flow:
                self.all_fwd = torch.cat(
                    self.all_fwd, 0
                )  # (len(self.meta['frames])*h*w, 2)
                self.all_bwd = torch.cat(
                    self.all_bwd, 0
                )  # (len(self.meta['frames])*h*w, 2)
                self.all_fwd_mask = torch.cat(
                    self.all_fwd_mask, 0
                )  # (len(self.meta['frames])*h*w, 1)
                self.all_bwd_mask = torch.cat(
                    self.all_bwd_mask, 0
                )  # (len(self.meta['frames])*h*w, 1)
            self.all_loss_weights = torch.cat(
                self.all_loss_weights, 0
            )  # (len(self.meta['frames])*h*w, 1)

        else:
            self.all_rays = torch.stack(
                self.all_rays, 0
            )  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(
                -1, *self.img_wh[::-1], 3
            )  # (len(self.meta['frames]),h,w,3)
            self.all_times = torch.stack(self.all_times, 0)
            self.all_loss_weights = torch.stack(self.all_loss_weights, 0).reshape(
                -1, *self.img_wh[::-1], 1
            )
            if self.depth_data:
                self.all_depths = torch.stack(self.all_depths, 0).reshape(
                    -1, *self.img_wh[::-1], 1
                )  # (len(self.meta['frames]),h,w,1)
            if self.bpixel_mask_data:
                self.bpixel_mask = self.bpixel_mask.reshape(-1, *self.img_wh[::-1], 1)
            if self.pixel_feat_data:
                self.all_pixel_feat = torch.stack(self.all_pixel_feat, 0).reshape(
                    -1, *self.img_wh[::-1], self.cfg.model.pixel_feat_dim
                )  # (len(self.meta['frames]),h,w,self.cfg.model.pixel_feat_dim)
                self.pixel_feat_mask = self.pixel_feat_mask.reshape(
                    -1, *self.img_wh[::-1], 1
                )

            if self.load_flow:
                self.all_fwd = torch.stack(self.all_fwd, 0).reshape(
                    -1, *self.img_wh[::-1], 2
                )  # (len(self.meta['frames]),h,w,2)
                self.all_bwd = torch.stack(self.all_bwd, 0).reshape(
                    -1, *self.img_wh[::-1], 2
                )  # (len(self.meta['frames]),h,w,2)
                self.all_fwd_mask = torch.stack(self.all_fwd_mask, 0).reshape(
                    -1, *self.img_wh[::-1], 1
                )  # (len(self.meta['frames]),h,w,1)
                self.all_bwd_mask = torch.stack(self.all_bwd_mask, 0).reshape(
                    -1, *self.img_wh[::-1], 1
                )  # (len(self.meta['frames]),h,w,1)

        all_imgs = copy.deepcopy(self.all_rgbs.reshape(-1, *self.img_wh[::-1], 3))

        self.all_times = self.time_scale * (self.all_times * 2.0 - 1.0)
        self.global_mean_rgb = torch.mean(all_imgs, dim=0)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:, :3]

    def world2ndc(self, points, lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)

    def __len__(self):
        return len(self.all_rgbs)

    def get_val_pose(self):
        """
        Get validation poses and times (NeRF-like rotating cameras poses).
        """
        render_poses = torch.stack(
            [
                pose_spherical(angle, -30.0, 4.0)
                for angle in np.linspace(-180, 180, 40 + 1)[:-1]
            ],
            0,
        )
        render_times = torch.linspace(0.0, 1.0, render_poses.shape[0]) * 2.0 - 1.0
        return render_poses, self.time_scale * render_times

    def get_val_rays(self):
        """
        Get validation rays and times (NeRF-like rotating cameras poses).
        """
        val_poses, val_times = self.get_val_pose()  # get valitdation poses and times
        rays_all = []  # initialize list to store [rays_o, rays_d]

        for i in range(val_poses.shape[0]):
            c2w = torch.FloatTensor(val_poses[i])
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)
            rays_all.append(rays)
        return rays_all, torch.FloatTensor(val_times)

    def __getitem__(self, idx):
        if self.split == "train":  # use data in the buffers
            sample = {
                "rays": self.all_rays[idx],
                "rgbs": self.all_rgbs[idx],
                "time": self.all_times[idx],
                "loss_weights": self.all_loss_weights[idx],
            }
        else:  # create data for each image separately
            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            time = self.all_times[idx]
            loss_weights = self.all_loss_weights[idx]
            sample = {
                "rays": rays,
                "rgbs": img,
                "time": time,
                "loss_weights": loss_weights,
            }

        if self.depth_data:
            sample["depths"] = self.all_depths[idx]
        if self.bpixel_mask_data:
            sample["bpixel_mask"] = self.bpixel_mask[idx]
        if self.pixel_feat_data:
            sample["pixel_feat"] = self.all_pixel_feat[idx]
            sample["pixel_feat_mask"] = self.pixel_feat_mask[idx]
        if self.load_flow:
            sample["fwd"] = self.all_fwd[idx]
            sample["bwd"] = self.all_bwd[idx]
            sample["fwd_mask"] = self.all_fwd_mask[idx]
            sample["bwd_mask"] = self.all_bwd_mask[idx]

        return sample

    def get_random_pose(self, batch_size, patch_size, batching="all_images"):
        """
        Apply Geometry Regularization from RegNeRF.
        This function randomly samples many patches from random poses.
        """
        n_patches = batch_size // (patch_size**2)

        N_random = self.random_rays.shape[0]
        # Sample images
        if batching == "all_images":
            idx_img = np.random.randint(0, N_random, size=(n_patches, 1))
        elif batching == "single_image":
            idx_img = np.random.randint(0, N_random)
            idx_img = np.full((n_patches, 1), idx_img, dtype=np.int)
        else:
            raise ValueError("Not supported batching type!")
        idx_img = torch.Tensor(idx_img).long()
        H, W = self.random_rays[0].shape[0], self.random_rays[0].shape[1]
        # Sample start locations
        x0 = np.random.randint(
            int(W // 4), int(W // 4 * 3) - patch_size + 1, size=(n_patches, 1, 1)
        )
        y0 = np.random.randint(
            int(H // 4), int(H // 4 * 3) - patch_size + 1, size=(n_patches, 1, 1)
        )
        xy0 = np.concatenate([x0, y0], axis=-1)
        patch_idx = xy0 + np.stack(
            np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing="xy"),
            axis=-1,
        ).reshape(1, -1, 2)

        patch_idx = torch.Tensor(patch_idx).long()
        # Subsample images
        out = self.random_rays[idx_img, patch_idx[..., 1], patch_idx[..., 0]]

        return out, self.random_times[idx_img]

    def get_new_pose_rays(self, pose, times, time):
        xs = list(np.zeros(times))
        ys = list(np.zeros(times))
        zs = list(np.zeros(times))

        rot_deg = 0.5
        rxs = (
            list(np.zeros(2))
            + list(np.ones(2) * -rot_deg)
            + list(np.zeros(2))
            + list(np.ones(2) * 1.5 * rot_deg)
            + list(np.zeros(2))
            + list(np.ones(2) * -2 * rot_deg)
            + list(np.ones(2) * 1.5 * rot_deg)
        )
        rys = (
            list(np.ones(2) * rot_deg)
            + list(np.zeros(2))
            + list(np.ones(2) * -1.5 * rot_deg)
            + list(np.zeros(2))
            + list(np.ones(2) * 2 * rot_deg)
            + list(np.zeros(2))
            + list(np.ones(2) * -1.5 * rot_deg)
        )
        rzs = list(np.zeros(times))

        if times == 0:  # useful for simulating fixed pose for all timesteps
            rays_all = []  # initialize list to store [rays_o, rays_d]
            poses = pose.reshape(1, pose.shape[0], pose.shape[1])
            time_a = time
            for i in range(1):
                c2w = torch.FloatTensor(poses[i])
                rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
                rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)
                rays_all.append(rays)

        for i in range(times):
            # x, y, z, rx, ry, rz = ...
            x, y, z, rx, ry, rz = xs[i], 0, zs[i], rxs[i], rys[i], rzs[i]
            c2w = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
            R_X = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, np.cos(rx * 3.1415 / 180), -np.sin(rx * 3.1415 / 180), 0.0],
                    [0.0, np.sin(rx * 3.1415 / 180), np.cos(rx * 3.1415 / 180), 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

            R_Y = np.array(
                [
                    [np.cos(ry * 3.1415 / 180), 0.0, np.sin(ry * 3.1415 / 180), 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [-np.sin(ry * 3.1415 / 180), 0.0, np.cos(ry * 3.1415 / 180), 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

            R_Z = np.array(
                [
                    [np.cos(rz * 3.1415 / 180), -np.sin(rz * 3.1415 / 180), 0.0, 0.0],
                    [np.sin(rz * 3.1415 / 180), np.cos(rz * 3.1415 / 180), 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

            if i == 0:
                pose = pose.detach().cpu().numpy()

            new_pose = R_X @ R_Y @ R_Z @ c2w @ pose[:4, :]

            new_time = time + (i + 1) * 0.00001

            new_pose = new_pose.reshape(
                1, new_pose.shape[0], new_pose.shape[1]
            )  # [:, :3]

            if i == 0:
                poses = np.concatenate(
                    (pose.reshape(1, pose.shape[0], pose.shape[1]), new_pose)
                )
                time_a = np.concatenate((time, new_time))
                time = new_time
                pose = new_pose[0]
            else:
                poses = np.concatenate((poses, new_pose))
                time_a = np.concatenate((time_a, new_time))
                pose = new_pose[0]
                time = new_time

        rays_all = []  # initialize list to store [rays_o, rays_d]
        for i in range(poses.shape[0]):
            c2w = torch.FloatTensor(poses[i])
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)
            rays_all.append(rays)

        return rays_all, torch.FloatTensor(time_a)
