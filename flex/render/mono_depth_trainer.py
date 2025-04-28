import math
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import sys
import time

import numpy as np
import torch
from trainer import Trainer

from flex.dataloader.ray_utils import get_ray_directions_lean, get_rays_lean, ids2pixel
from flex.render.render import OctreeRender_trilinear_fast as renderer
from flex.render.render import evaluation
from flex.render.util.Reg import TVLoss
from flex.render.util.Sampling import cal_n_samples
from flex.render.util.util import (
    N_to_reso,
    get_fwd_bwd_cam2cams,
    get_pred_flow,
)


class MonoDepthTrainer(Trainer):
    def __init__(
        self,
        model,
        cfg,
        reso_cur,
        train_dataset,
        test_dataset,
        summary_writer,
        logfolder,
        device,
        world2hex=None,
        image_bound=None,
    ):
        super().__init__(
            model,
            cfg,
            reso_cur,
            train_dataset,
            test_dataset,
            summary_writer,
            logfolder,
            device,
        )

        print(f"total frames {self.total_frames}")

    def train(self):
        torch.cuda.empty_cache()

        # load the training and testing dataset and other settings.
        train_dataset = self.train_dataset
        test_dataset = self.test_dataset
        model = self.model
        self.depth_data = test_dataset.depth_data
        summary_writer = self.summary_writer
        reso_cur = self.reso_cur

        ndc_ray = train_dataset.ndc_ray  # if the rays are in NDC
        white_bg = test_dataset.white_bg  # if the background is white

        # Calculate the number of samples for each ray based on the current resolution.
        nSamples = min(
            self.cfg.model.nSamples,
            cal_n_samples(reso_cur, self.cfg.model.step_ratio),
        )
        self.nSamples = nSamples

        # Filter the rays based on the bbox
        # if (self.cfg.data.datasampler_type == "rays") and (ndc_ray is False):
        #     allrays, allrgbs, alltimes = (
        #         train_dataset.all_rays,
        #         train_dataset.all_rgbs,
        #         train_dataset.all_times,
        #     )
        #     if self.depth_data:
        #         alldepths = train_dataset.all_depths
        #     else:
        #         alldepths = None
        #     if not self.cfg.local_models:
        #         allrays, allrgbs, alltimes, alldepths = model.filtering_rays(
        #             allrays, allrgbs, alltimes, alldepths, bbox_only=True
        #         )
        #         train_dataset.all_rays = allrays
        #         train_dataset.all_rgbs = allrgbs
        #         train_dataset.all_times = alltimes
        #         train_dataset.all_depths = alldepths

        # initialize the data sampler
        self.init_sampler(train_dataset)
        # precompute the voxel upsample list
        self.get_voxel_upsample_list()

        # Initialiaze TV loss on planse
        tvreg_s = TVLoss()  # TV loss on the spatial planes
        tvreg_s_t = TVLoss(
            1.0, self.cfg.model.TV_t_s_ratio
        )  # TV loss on the spatial-temporal planes

        PSNRs, PSNRs_test = [], [0]
        torch.cuda.empty_cache()

        # Initialize the optimizer
        grad_vars = model.get_optparam_groups(self.cfg.optim)
        optimizer = torch.optim.Adam(
            grad_vars, betas=(self.cfg.optim.beta1, self.cfg.optim.beta2)
        )
        if self.cfg.local.progressive_opt and self.cfg.local_models:
            self.optimizers_hex = []
            self.optimizers_hex.append(optimizer)
            self.hexplanes = []
            self.hexplanes.append(model)

        # # Initialize pose optimization
        # if self.optimize_poses and not self.cfg.local.progressive_opt:
        #     self.poses_rot = []
        #     self.poses_t = []
        #     self.optimizers_Rot = []
        #     self.optimizers_T = []
        #     # include test poses
        #     for i in range(self.active_frames_bounds[1] - self.active_frames_bounds[0]):
        #         pose_rot = torch.nn.parameter.Parameter(
        #             self.all_poses[i, :3, :3], requires_grad=True
        #         )
        #         self.poses_rot.append(pose_rot)
        #         pose_t = torch.nn.parameter.Parameter(
        #             self.all_poses[i, :3, 3], requires_grad=True
        #         )
        #         self.poses_t.append(pose_t)
        #         optimizer_Rot = torch.optim.Adam(
        #             [self.poses_rot[-1]],
        #             betas=(self.cfg.optim.beta1, self.cfg.optim.beta2),
        #             lr=self.cfg.optim.lr_R_init,
        #         )
        #         self.optimizers_Rot.append(optimizer_Rot)
        #         optimizer_T = torch.optim.Adam(
        #             [self.poses_t[-1]],
        #             betas=(self.cfg.optim.beta1, self.cfg.optim.beta2),
        #             lr=self.cfg.optim.lr_T_init,
        #         )
        #         self.optimizers_T.append(optimizer_T)

        total_iteration = 0
        last_add_iter = 0
        self.local_iteration = [0]
        iter_time_acc = 0.0
        while self.training:
            # append frames if needed
            start_iter_time = time.time()

            if self.cfg.local_models and self.cfg.local.progressive_opt:

                create_new_model, last_add_iter = self.check_new_model(
                    total_iteration, last_add_iter, train_dataset
                )
                if (
                    not self.cfg.optim.optimize_poses
                    and self.cfg.local.add_frames_every == 0
                ):
                    while not self.refine_model:
                        create_new_model, last_add_iter = self.check_new_model(
                            total_iteration, last_add_iter, train_dataset
                        )
                if create_new_model:
                    if self.active_frames_bounds[1] < self.total_frames:
                        add_last_iter, reso_cur = self.create_new_model(
                            total_iteration, train_dataset
                        )
                        model = self.hexplanes[-1]
                    else:
                        self.training = False
                        self.image_bounds.append(self.image_bound)
                        torch.save(
                            model,
                            f"{self.logfolder}/{self.cfg.expname}_{(len(self.local_iteration)-1)}.th",
                        )
                        break

            # Sample data
            (
                rays_train,
                rgb_train,
                frame_time,
                depth,
                gt_fwd,
                gt_bwd,
                fwd_mask,
                bwd_mask,
                ray_idx,
                img_idxs,
                train_test_poses,
                tool_mask,
                spec_mask,
            ) = self.sample_data(train_dataset, test_dataset, self.local_iteration[-1])
            # Render the rgb values of rays
            # last item is sigma --> try again when having more memory
            rgb_map, alphas_map, depth_map, z_vals, xyz_sampled, weights = renderer(
                rays_train,
                frame_time,
                model,
                chunk=self.cfg.optim.batch_size,
                N_samples=nSamples,
                white_bg=white_bg,
                ndc_ray=ndc_ray,
                device=self.device,
                is_train=True,
            )

            # Calculate the loss
            loss = torch.mean((rgb_map - rgb_train) ** 2)
            total_loss = self.cfg.model.rgb_loss_weight * loss

            # Calculate the learning rate decay factor
            if self.cfg.local.progressive_opt and not self.refine_model:
                lr_factor = 1
            else:
                lr_factor = self.get_lr_decay_factor(self.local_iteration[-1])
            # Calculate Temp decay rate (influence of uncertain supervision signals)
            # Temp = 1. / (10 ** (self.local_iteration[-1] // (self.decay_iteration * 45))) # originally 1000 instead of 30 in most inner bracket (last value)
            if (
                self.cfg.local.progressive_opt and not self.refine_model
            ) or total_iteration == 0:
                Temp = 1.0
                summary_writer.add_scalar(
                    "train/Temp",
                    Temp,
                    global_step=total_iteration,
                )
            else:
                Temp = 1.0 / (
                    10
                    ** (
                        self.local_iteration[-1]
                        // (self.n_iters // self.cfg.optim.temp_decline)
                    )
                )
                summary_writer.add_scalar(
                    "train/Temp",
                    Temp,
                    global_step=total_iteration,
                )

            # Decay the learning rate.
            if self.cfg.local_models and self.cfg.local.progressive_opt:
                self.update_optimizers()

            # regularization
            # TV loss on the density planes
            if self.cfg.model.TV_weight_density > 0:
                TV_weight_density = lr_factor * self.cfg.model.TV_weight_density
                loss_tv = model.TV_loss_density(tvreg_s, tvreg_s_t) * TV_weight_density
                total_loss = total_loss + loss_tv
                summary_writer.add_scalar(
                    "train/reg_tv_density",
                    loss_tv.detach().item(),
                    global_step=total_iteration,
                )

            # TV loss on the appearance planes
            if self.cfg.model.TV_weight_app > 0:
                TV_weight_app = lr_factor * self.cfg.model.TV_weight_app
                loss_tv = model.TV_loss_app(tvreg_s, tvreg_s_t) * TV_weight_app
                total_loss = total_loss + loss_tv
                summary_writer.add_scalar(
                    "train/reg_tv_app",
                    loss_tv.detach().item(),
                    global_step=total_iteration,
                )

            # L1 loss on the density planes
            if self.cfg.model.L1_weight_density > 0:
                L1_weight_density = lr_factor * self.cfg.model.L1_weight_density
                loss_l1 = model.L1_loss_density() * L1_weight_density
                total_loss = total_loss + loss_l1
                summary_writer.add_scalar(
                    "train/reg_l1_density",
                    loss_l1.detach().item(),
                    global_step=total_iteration,
                )

            # L1 loss on the appearance planes
            if self.cfg.model.L1_weight_app > 0:
                L1_weight_app = lr_factor * self.cfg.model.L1_weight_app
                loss_l1 = model.L1_loss_app() * L1_weight_app
                total_loss = total_loss + loss_l1
                summary_writer.add_scalar(
                    "train/reg_l1_app",
                    loss_l1.detach().item(),
                    global_step=total_iteration,
                )

            if self.cfg.model.depth_loss and self.cfg.model.urf_depth_loss_weight > 0:
                """Lidar losses from Urban Radiance Fields (Rematas et al., 2022).
                Args:
                    weights: Weights predicted for each sample.
                    depth: Ground truth depth of rays.
                    depth_map: Depth prediction from the network.
                    z_vals: Sampling distances along rays.
                    sigma: Uncertainty around depth values.
                """
                sigma = (
                    10.0 / self.train_dataset.original_depth_scale
                )  # 0.01 # default should be ca. 1cm
                if self.cfg.local.progressive_opt and self.refine_model:
                    sigma = sigma * self.cfg.optim.lr_decay_target_ratio ** (
                        self.local_iteration[-1] / self.n_iters
                    )
                URF_SIGMA_SCALE_FACTOR = 3.0
                depth_mask = depth > 0
                target_dist_scale = 1 / 12000  # avoid exp() explosion

                # Expected depth loss
                expected_depth_loss = (depth - depth_map.unsqueeze(-1)) ** 2

                # Line of sight losses
                target_distribution = torch.distributions.normal.Normal(
                    0.0, sigma / URF_SIGMA_SCALE_FACTOR
                )
                depth = depth[:, None]
                dists = z_vals.reshape(1, z_vals.shape[-1], 1).repeat(
                    depth.shape[0], 1, 1
                )
                weights = weights[..., None]

                line_of_sight_loss_near_mask = torch.logical_and(
                    dists <= depth + sigma, dists >= depth - sigma
                )
                line_of_sight_loss_near = (
                    weights
                    - target_dist_scale
                    * torch.exp(target_distribution.log_prob(dists - depth))
                ) ** 2
                line_of_sight_loss_near = (
                    line_of_sight_loss_near_mask * line_of_sight_loss_near
                ).sum(-2)
                line_of_sight_loss_empty_mask = dists < depth - sigma
                line_of_sight_loss_empty = (
                    line_of_sight_loss_empty_mask * weights**2
                ).sum(-2)
                line_of_sight_loss = line_of_sight_loss_near + line_of_sight_loss_empty
                urf_depth_loss = torch.mean(
                    (expected_depth_loss + line_of_sight_loss) * depth_mask
                )
                total_loss += urf_depth_loss * self.cfg.model.urf_depth_loss_weight

                summary_writer.add_scalar(
                    "train/urban_rad_field_depth_loss",
                    urf_depth_loss.detach().item(),
                    global_step=total_iteration,
                )
                weights = weights.reshape(weights.shape[0], weights.shape[1])

            # Optical flow loss
            # start_time_train_optical = time.time()
            if (
                self.cfg.data.flow_data
                and (self.cfg.model.optical_flow_loss_weight > 0)
                and (
                    (not self.refine_model)
                    or (self.local_iteration[-1] / self.n_iters) < 0.2
                )
            ):
                W, H = train_dataset.img_wh
                focal = train_dataset.focal

                # mask out flow into test images
                if not train_test_poses:
                    mask_test_prev, mask_test_post = train_dataset.prev_test[
                        img_idxs
                    ].to(self.device), train_dataset.post_test[img_idxs].to(self.device)
                    mask_test_prev = ~(mask_test_prev == 1)
                    mask_test_post = ~(mask_test_post == 1)

                    if self.cfg.data.datasampler_type == "images":
                        mask_test_prev = mask_test_prev.reshape(
                            self.cfg.optim.num_s_imgs, 1
                        )
                        mask_test_prev = mask_test_prev.repeat(
                            1, self.cfg.optim.batch_size // self.cfg.optim.num_s_imgs
                        )
                        mask_test_post = mask_test_post.reshape(
                            self.cfg.optim.num_s_imgs, 1
                        )
                        mask_test_post = mask_test_post.repeat(
                            1, self.cfg.optim.batch_size // self.cfg.optim.num_s_imgs
                        )

                    actual_img_idx = torch.tensor(self.train_images_idxs[img_idxs]).to(
                        self.device
                    )
                    img_idx_prev = actual_img_idx - 1
                    img_idx_post = actual_img_idx + 1

                else:
                    actual_img_idx = (img_idxs * 8).to(
                        self.device
                    )  # map to global_img_idxs
                    img_idx_prev = actual_img_idx - 1
                    img_idx_post = actual_img_idx + 1

                if self.cfg.data.datasampler_type == "rays":
                    img_idx_prev[img_idx_prev < 0] = (
                        0  # set to 0 --> note this will be filtered out based on masks anyways
                    )
                    img_idx_post[img_idx_post > self.total_frames - 1] = (
                        self.total_frames - 1
                    )  # set to max value --> note this will be filtered out based on masks anyways
                else:
                    img_idx_prev[img_idx_prev <= 0] = 0
                    img_idx_post[img_idx_post > (self.total_frames - 1)] = (
                        self.total_frames - 1
                    )

                if self.cfg.local.progressive_opt:
                    progressive_mask = img_idx_post < (len(self.poses_rot) - 1)
                    progressive_mask = progressive_mask.reshape(1, -1)
                    img_idx_post[img_idx_post > (len(self.poses_rot) - 1)] = (
                        len(self.poses_rot) - 1
                    )  # placeholder --> need to be filtered out

                    if self.cfg.data.datasampler_type == "images":
                        progressive_mask = progressive_mask.reshape(
                            self.cfg.optim.num_s_imgs, 1
                        )
                        progressive_mask = progressive_mask.repeat(
                            1, self.cfg.optim.batch_size // self.cfg.optim.num_s_imgs
                        )
                        img_idx_prev = img_idx_prev[..., None]
                        img_idx_post = img_idx_post[..., None]
                        actual_img_idx = actual_img_idx[..., None]
                    else:
                        poses_prev = self.get_cam2world(img_idx_prev)
                        poses_post = self.get_cam2world(img_idx_post)
                        poses = self.get_cam2world(actual_img_idx)

                else:
                    poses_prev = self.all_poses.to(self.device)[img_idx_prev]
                    poses_post = self.all_poses.to(self.device)[img_idx_post]
                    poses = self.all_poses.to(self.device)[actual_img_idx]

                i, j = ids2pixel(W, H, ray_idx)
                px_original = torch.stack([i, j], dim=-1)

                ray_idx_adj = ray_idx % (H * W)
                i, j = ids2pixel(W, H, ray_idx_adj)
                directions = get_ray_directions_lean(
                    i,
                    j,
                    [train_dataset.focal, train_dataset.focal],
                    [train_dataset.cx, train_dataset.cy],
                )

                # mask out flow into test frames
                if self.cfg.data.tool_mask:
                    # mask out tool pixels
                    fwd_mask = torch.logical_and(fwd_mask, tool_mask)
                    bwd_mask = torch.logical_and(bwd_mask, tool_mask)

                if self.cfg.data.datasampler_type == "images":
                    fwd_mask = fwd_mask.view(
                        self.cfg.optim.num_s_imgs,
                        self.cfg.optim.batch_size // self.cfg.optim.num_s_imgs,
                    )
                    bwd_mask = bwd_mask.view(
                        self.cfg.optim.num_s_imgs,
                        self.cfg.optim.batch_size // self.cfg.optim.num_s_imgs,
                    )
                    gt_fwd = gt_fwd.view(
                        self.cfg.optim.num_s_imgs,
                        self.cfg.optim.batch_size // self.cfg.optim.num_s_imgs,
                        2,
                    )
                    gt_bwd = gt_bwd.view(
                        self.cfg.optim.num_s_imgs,
                        self.cfg.optim.batch_size // self.cfg.optim.num_s_imgs,
                        2,
                    )
                else:
                    fwd_mask = fwd_mask.view(1, -1)
                    bwd_mask = bwd_mask.view(1, -1)
                    gt_fwd = gt_fwd.view(-1, 1, 2)
                    gt_bwd = gt_bwd.view(-1, 1, 2)

                if not train_test_poses and not self.optimize_poses:
                    fwd_mask = torch.logical_and(fwd_mask, mask_test_post)
                    bwd_mask = torch.logical_and(bwd_mask, mask_test_prev)

                if self.cfg.local.progressive_opt:
                    # mask out not seen frames yet
                    fwd_mask = torch.logical_and(fwd_mask, progressive_mask)

                if self.cfg.model.opt_from_depth:
                    if self.cfg.local.progressive_opt:
                        cam2world = self.get_cam2world()
                    else:
                        cam2world = self.all_poses[:, :3, :4].to(self.device)
                        actual_img_idx = actual_img_idx.to(self.device)

                    actual_img_idx = actual_img_idx.reshape(-1)
                    poses_post, poses_prev = get_fwd_bwd_cam2cams(
                        cam2world, actual_img_idx
                    )

                if self.cfg.data.datasampler_type == "images":
                    # use model prediction
                    pts = directions.to(self.device)[None, ...] * depth_map[..., None]
                    pts = pts.view(self.cfg.optim.num_s_imgs, -1, 3)
                    px_original = px_original.reshape(pts.shape[0], -1, 2).to(
                        self.device
                    )
                else:
                    pts = (
                        directions.to(self.device)[None, ...]
                        * depth_map[..., None].clone().detach()
                    )
                    px_original = px_original.reshape(pts.shape[0], -1, 2).to(
                        self.device
                    )

                pred_fwd_flow, new_px_fwd = get_pred_flow(
                    pts,
                    px_original,
                    poses_post,
                    focal,
                    [train_dataset.cx, train_dataset.cy],
                )
                pred_bwd_flow, new_px_bwd = get_pred_flow(
                    pts,
                    px_original,
                    poses_prev,
                    focal,
                    [train_dataset.cx, train_dataset.cy],
                )

                if self.cfg.model.optical_flow_loss_weight > 0:
                    flow_loss_arr = (
                        torch.sum(torch.abs(pred_bwd_flow - gt_bwd), dim=-1) * bwd_mask
                    )
                    flow_loss_arr += (
                        torch.sum(torch.abs(pred_fwd_flow - gt_fwd), dim=-1) * fwd_mask
                    )
                    flow_loss_arr[
                        flow_loss_arr
                        > torch.quantile(flow_loss_arr, 0.9, dim=1)[..., None]
                    ] = 0
                    flow_loss = (flow_loss_arr).mean() / ((W + H) / 2)

                    total_loss = total_loss + (
                        flow_loss * self.cfg.model.optical_flow_loss_weight * Temp
                    )
                    summary_writer.add_scalar(
                        "train/flow_loss", flow_loss, global_step=total_iteration
                    )

            # Optimization
            if self.cfg.local.progressive_opt and not self.refine_model:
                self.n_iters = self.local_iteration[-1] + 1  # placeholder

            if self.optimize_poses and train_test_poses:
                # optimise only poses
                for opt in range(len(self.optimizers_Rot)):
                    if (
                        self.pose_linked_hex[opt] == len(self.local_iteration) - 1
                        and self.local_iteration[-1] < self.n_iters
                    ):
                        self.optimizers_Rot[opt].zero_grad()
                        self.optimizers_T[opt].zero_grad()
                total_loss.backward()
                for opt in range(len(self.optimizers_Rot)):
                    if (
                        self.pose_linked_hex[opt] == len(self.local_iteration) - 1
                        and self.local_iteration[-1] < self.n_iters
                    ):
                        self.optimizers_Rot[opt].step()
                        self.optimizers_T[opt].step()
            elif self.optimize_poses:
                # optimise poses
                for opt in range(len(self.optimizers_Rot)):
                    if self.cfg.local_models:
                        if (
                            self.pose_linked_hex[opt] == len(self.local_iteration) - 1
                            and self.local_iteration[-1] < self.n_iters
                        ):
                            self.optimizers_Rot[opt].zero_grad()
                            self.optimizers_T[opt].zero_grad()
                    else:
                        self.optimizers_Rot[opt].zero_grad()
                        self.optimizers_T[opt].zero_grad()
                # optimise model
                if self.cfg.local.progressive_opt and self.cfg.local_models:
                    self.optimizers_hex[-1].zero_grad()
                else:
                    optimizer.zero_grad()
                total_loss.backward()
                if self.cfg.local.progressive_opt and self.cfg.local_models:
                    self.optimizers_hex[-1].step()
                else:
                    optimizer.step()

                for opt in range(len(self.optimizers_Rot)):
                    if self.cfg.local_models:
                        if (
                            self.pose_linked_hex[opt] == len(self.local_iteration) - 1
                            and self.local_iteration[-1] < self.n_iters
                        ):
                            self.optimizers_Rot[opt].step()
                            self.optimizers_T[opt].step()
                    else:
                        self.optimizers_Rot[opt].step()
                        self.optimizers_T[opt].step()
            else:
                if self.cfg.local.progressive_opt and self.cfg.local_models:
                    self.optimizers_hex[-1].zero_grad()
                    total_loss.backward()
                    self.optimizers_hex[-1].step()
                else:
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

            loss = loss.detach().item()
            PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
            summary_writer.add_scalar(
                "train/PSNR", PSNRs[-1], global_step=total_iteration
            )
            summary_writer.add_scalar("train/mse", loss, global_step=total_iteration)

            iter_time_acc += time.time() - start_iter_time
            if total_iteration % self.cfg.systems.progress_refresh_rate == 0:
                print(
                    f"Iteration = {total_iteration}, train_psnr = {float(np.mean(PSNRs)):.2f}, test_psnr = {float(np.mean(PSNRs_test)):.2f}, mse = {loss:.6f}, iters/sec: {(self.cfg.systems.progress_refresh_rate/iter_time_acc):.2f}"
                )
                PSNRs = []
                iter_time_acc = 0.0

            elif not self.cfg.local.progressive_opt:
                for param_group in optimizer.param_groups:
                    if self.cfg.optim.lr_decay_type == "cyclic":
                        last_epoch = self.local_iteration[-1] - 1  # -1
                        max_lr = param_group["lr_org"]
                        base_lr = max_lr * self.cfg.optim.lr_decay_target_ratio
                        cycle = np.floor(
                            1 + last_epoch / (2 * self.cfg.optim.cyc_step_size)
                        )
                        x = np.abs(
                            float(last_epoch) / self.cfg.optim.cyc_step_size
                            - 2 * cycle
                            + 1
                        )
                        if self.cfg.optim.cyc_mode == "triangular":
                            lr = base_lr + (
                                self.cfg.optim.lr_decay_target_ratio
                            ) * np.maximum(0, (1 - x))
                        elif self.cfg.optim.cyc_mode == "triangular2":
                            lr = base_lr + (
                                self.cfg.optim.lr_decay_target_ratio
                            ) * np.maximum(0, (1 - x)) / float(2 ** (cycle - 1))
                        elif self.cfg.optim.cyc_mode == "exp_range":
                            lr = base_lr + (
                                self.cfg.optim.lr_decay_target_ratio
                            ) * np.maximum(0, (1 - x)) * (
                                self.cfg.optim.cyc_gamma ** (last_epoch)
                            )
                        param_group["lr"] = lr
                    else:
                        param_group["lr"] = param_group["lr_org"] * lr_factor

            # Evaluation for every self.cfg.systems.vis_every steps.
            if (
                total_iteration % self.cfg.systems.vis_every
                == self.cfg.systems.vis_every - 1
                and self.cfg.data.N_vis != 0
            ) or (
                self.local_iteration[-1] == (self.n_iters - 1)
                and self.cfg.local.progressive_opt
                and self.cfg.local_models
                and self.refine_model
            ):
                # select poses for rendering
                if self.cfg.local_models and self.cfg.local.progressive_opt:
                    poses = self.get_cam2world()
                    poses_R = poses[:, :3, :3]
                    poses_T = poses[:, :3, 3]
                    # save current model
                    torch.save(
                        model,
                        f"{self.logfolder}/{self.cfg.expname}_{(len(self.local_iteration)-1)}.th",
                    )
                else:
                    poses_R = self.all_poses[:, :3, :3]
                    poses_T = self.all_poses[:, :3, 3]

                PSNRs_test = evaluation(
                    test_dataset,
                    model,
                    self.cfg,
                    f"{self.logfolder}/imgs_vis/",
                    prefix=f"{total_iteration:06d}_",
                    white_bg=white_bg,
                    N_samples=nSamples,
                    ndc_ray=ndc_ray,
                    device=self.device,
                    compute_extra_metrics=False,
                    image_bound=self.image_bound,
                    world2hexs=self.world2hexs if self.cfg.local_models else None,
                    num_train_images=(
                        len(self.train_images_idxs)
                        if self.cfg.local_models
                        else len(train_dataset.poses)
                    ),
                    poses_rot=poses_R,
                    poses_t=poses_T,
                    blending_weights=(
                        self.blending_weights
                        if self.cfg.local.progressive_opt
                        else None
                    ),
                    hexplanes=(
                        self.hexplanes if self.cfg.local.progressive_opt else None
                    ),
                    modelsdirectory=f"{self.logfolder}/{self.cfg.expname}",
                )
                summary_writer.add_scalar(
                    "test/psnr", np.mean(PSNRs_test), global_step=total_iteration
                )

                torch.cuda.synchronize()

            # Calculate the emptiness voxel.
            if (
                self.local_iteration[-1] in self.update_emptymask_list
                and not self.cfg.model.no_emptyMask
            ):
                if (
                    reso_cur[0] * reso_cur[1] * reso_cur[2] < 256**3
                ):  # update volume resolution
                    reso_mask = reso_cur
                model.updateEmptyMask(tuple(reso_mask))

                torch.cuda.empty_cache()
                print("Updated empty mask!")

            # Upsample the volume grid.
            if self.local_iteration[-1] in self.upsample_list:
                if self.cfg.model.upsampling_type == "aligned":
                    reso_cur = [reso_cur[i] * 2 - 1 for i in range(len(reso_cur))]
                else:
                    N_voxel = self.N_voxel_list.pop(0)
                    reso_cur = N_to_reso(
                        N_voxel, model.aabb, self.cfg.model.nonsquare_voxel
                    )
                time_grid = self.Time_grid_list.pop(0)
                nSamples = min(
                    self.cfg.model.nSamples,
                    cal_n_samples(reso_cur, self.cfg.model.step_ratio),
                )
                model.upsample_volume_grid(reso_cur, time_grid)

                grad_vars = model.get_optparam_groups(self.cfg.optim, 1.0)
                if self.cfg.local.progressive_opt:
                    self.optimizers_hex[-1] = torch.optim.Adam(
                        grad_vars, betas=(self.cfg.optim.beta1, self.cfg.optim.beta2)
                    )
                else:
                    optimizer = torch.optim.Adam(
                        grad_vars, betas=(self.cfg.optim.beta1, self.cfg.optim.beta2)
                    )

                torch.cuda.empty_cache()

            # save model checkpoint
            if (
                self.local_iteration[-1] % self.cfg.systems.save_ckpt_every == 0
                and self.local_iteration[-1] != 0
            ):
                if self.cfg.local_models:
                    torch.save(
                        model,
                        f"{self.logfolder}/{self.cfg.expname}_{(len(self.local_iteration)-1)}.th",
                    )
                    if self.cfg.local.progressive_opt:
                        # Save rotation and translation matrices:
                        torch.save(
                            torch.stack(list(self.poses_rot), 0),
                            f"{self.logfolder}/{self.cfg.expname}_poses_rot.th",
                        )
                        torch.save(
                            torch.stack(list(self.poses_t), 0),
                            f"{self.logfolder}/{self.cfg.expname}_poses_t.th",
                        )
                        # save image bounds and world2hex coordinates:
                        torch.save(
                            torch.tensor(self.image_bounds),
                            f"{self.logfolder}/{self.cfg.expname}_image_bounds.th",
                        )
                        torch.save(
                            torch.stack(list(self.world2hexs), 0),
                            f"{self.logfolder}/{self.cfg.expname}_world2hexs.th",
                        )
                        # save blending weights:
                        torch.save(
                            torch.stack(list(self.blending_weights), 0),
                            f"{self.logfolder}/{self.cfg.expname}_blending_weights.th",
                        )
                else:
                    torch.save(model, f"{self.logfolder}/{self.cfg.expname}.th")
                    # torch.save(model.state_dict(), f"{self.logfolder}/{self.cfg.expname}.th")

            # update iter count
            if not (self.cfg.local.progressive_opt and (not self.refine_model)):
                self.local_iteration[-1] += 1
            total_iteration += 1

            if not (self.cfg.local_models and self.cfg.local.progressive_opt):
                if self.n_iters <= self.local_iteration[-1]:
                    self.training = False

            # update MLP progress bar for coarse to fine Positional Encoding
            if self.cfg.model.barf_c2f is not None and not (
                self.cfg.model.barf_c2f[-1] == 0.0
            ):
                if not self.refine_model:
                    model.density_regressor.progress = 0.001
                    model.app_regressor.progress = 0.001
                    if (
                        self.cfg.model.predict_flow
                        and self.cfg.data.flow_data
                        and (not self.cfg.model.opt_from_depth)
                        and (not self.cfg.model.no_sep_sflow_mlp)
                    ):
                        model.sflow_regressor.progress = 0.001
                else:
                    model.density_regressor.progress = (
                        self.local_iteration[-1] / self.n_iters
                    )
                    model.app_regressor.progress = (
                        self.local_iteration[-1] / self.n_iters
                    )
                    if (
                        self.cfg.model.predict_flow
                        and self.cfg.data.flow_data
                        and (not self.cfg.model.opt_from_depth)
                        and (not self.cfg.model.no_sep_sflow_mlp)
                    ):
                        model.sflow_regressor.progress = (
                            self.local_iteration[-1] / self.n_iters
                        )
                weights_PE = model.density_regressor.get_PE_weights()
                summary_writer.add_scalar(
                    "train/density_fea_pe_c2f_weights",
                    np.mean(weights_PE.detach().cpu().numpy()),
                    global_step=total_iteration,
                )

    def sample_data(self, train_dataset, test_dataset, iteration):
        """
        Sample a batch of data from the dataset.

        Possible sample schemes:
        # sample rays: shuffle all the rays of training dataset and sampled a batch of rays from them.
        # sample images: randomly pick one image from the training dataset and sample a batch of rays from all the rays of the image.
        # Sample hierch: hierarchical sampling from dyNeRF: hierachical sampling involves three stages of samplings.

        """
        train_depth = None
        tool_mask = None
        gt_fwd = None
        gt_bwd = None
        fwd_mask = None
        bwd_mask = None
        # loss_weights = None
        train_test_poses = False
        spec_mask = None

        # sample images: randomly pick one image from the training dataset and sample a batch of rays from all the rays of the image.
        if self.cfg.data.datasampler_type == "images":
            if self.cfg.local_models:
                active_test_mask = self.test_mask[
                    self.active_frames_bounds[0] : self.active_frames_bounds[1]
                ]
            else:
                active_test_mask = self.test_mask
            if (
                self.cfg.local_models
                and self.cfg.local.progressive_opt
                and self.optimize_poses
            ):
                test_ratio = active_test_mask.mean()
                train_test_poses = test_ratio > random.uniform(0, 1)
            else:
                train_test_poses = False

            if train_test_poses:
                actual_test_frames = np.arange(0, test_dataset.poses.shape[0]) * 8
                active_test_frames_idxs = actual_test_frames[
                    torch.logical_and(
                        torch.tensor(actual_test_frames)
                        >= self.active_frames_bounds[0],
                        torch.tensor(actual_test_frames) < self.active_frames_bounds[1],
                    )
                ]
                # map to test dataset idxs
                active_test_frames_idxs = active_test_frames_idxs // 8
                img_idxs = torch.randint(
                    0,
                    len(active_test_frames_idxs),
                    (self.cfg.optim.num_s_imgs,),
                    dtype=torch.int64,
                )

                if not self.refine_model:  # Local Optimization
                    if (
                        self.cfg.optim.num_s_imgs > 5
                        and len(active_test_frames_idxs) > 4
                    ):
                        # enforce last 5 frames to be in batch for local optimization
                        img_idxs[:2] = active_test_frames_idxs[-1].item()
                        img_idxs[2:4] = active_test_frames_idxs[-1].item() - 1
                        img_idxs[4] = active_test_frames_idxs[-1].item() - 2
                        img_idxs[5] = active_test_frames_idxs[-1].item() - 3

                    elif self.cfg.optim.num_s_imgs == 1:
                        # enforce last frame to be only item in batch for local optimization
                        img_idxs[:1] = active_test_frames_idxs[-1].item()

                data = test_dataset[img_idxs]
                global_img_idxs = img_idxs * 8
                W, H = train_dataset.img_wh
                select_inds = torch.randperm(int(H * W))[: self.cfg.optim.batch_size]
                all_ray_idxs = select_inds
                i, j = ids2pixel(W, H, all_ray_idxs)
                directions = get_ray_directions_lean(
                    i,
                    j,
                    [test_dataset.focal, test_dataset.focal],
                    [test_dataset.cx, test_dataset.cy],
                )

                cam2world = self.get_cam2world(
                    global_img_idxs.reshape(self.cfg.optim.num_s_imgs, 1)
                )
                cam2hex = (
                    cam2world.clone().detach()
                )  # do not detach when using gradients for pose optim
                cam2hex[:, :3, 3] += self.world2hexs[-1]

                rays_train = get_rays_lean(
                    directions.to(self.device),
                    cam2hex.repeat_interleave(
                        directions.shape[0] // self.cfg.optim.num_s_imgs, dim=0
                    ),
                ).reshape(-1, 6)

                ray_idxs_repeated = torch.arange(
                    0, self.cfg.optim.num_s_imgs
                ).repeat_interleave(
                    self.cfg.optim.batch_size // self.cfg.optim.num_s_imgs, dim=0
                ) * (
                    W * H
                )
                select_inds += ray_idxs_repeated

                rgb_train, frame_time = (
                    data["rgbs"].to(self.device).view(-1, 3),
                    data["time"],
                )
                spec_mask = data["spec_mask"].to(self.device).view(-1, 1)[select_inds]

                rgb_train = rgb_train[select_inds]
                frame_time = frame_time.view(-1, 1)[select_inds]

                if self.depth_data:
                    train_depth = (
                        data["depths"].to(self.device).view(-1, 1)[select_inds]
                    )
                if self.train_dataset.load_tool_mask:
                    tool_mask = (
                        data["tool_mask"].to(self.device).view(-1, 1)[select_inds]
                    )
                if self.cfg.data.flow_data:
                    gt_fwd = data["fwd"].to(self.device).view(-1, 2)[select_inds]
                    gt_bwd = data["bwd"].to(self.device).view(-1, 2)[select_inds]
                    fwd_mask = data["fwd_mask"].to(self.device).view(-1, 1)[select_inds]
                    bwd_mask = data["bwd_mask"].to(self.device).view(-1, 1)[select_inds]

                ray_idx = select_inds
            else:
                img_i = self.sampler.nextids()
                if len(img_i) < self.cfg.optim.num_s_imgs:
                    repeats = math.ceil(self.cfg.optim.num_s_imgs / len(img_i))
                    img_i = img_i.repeat(repeats)[: self.cfg.optim.num_s_imgs]

                if self.cfg.local_models:
                    img_i += self.image_bound[0]

                if self.cfg.local.progressive_opt:
                    if (
                        not self.refine_model and self.optimize_poses
                    ):  # Local Optimization
                        if (
                            self.cfg.optim.num_s_imgs > 5
                            and (self.image_bound[-1] - self.image_bound[0]) > 4
                        ):
                            # enforce last 5 frames to be in batch for coarse optimization
                            img_i[:2] = self.image_bound[-1] - 1
                            img_i[2:4] = self.image_bound[-1] - 2
                            img_i[4] = self.image_bound[-1] - 3
                            img_i[5] = self.image_bound[-1] - 4
                        elif self.cfg.optim.num_s_imgs == 1:
                            # enforce last frame to be only item in batch for local optimization
                            img_i[:1] = self.image_bound[-1] - 1

                    data = train_dataset[img_i]

                    train_img_idxs = img_i
                    global_img_idxs = torch.tensor(
                        self.train_images_idxs[train_img_idxs]
                    )  # map from train img idxs to global img idxs
                    img_idxs = train_img_idxs
                    W, H = train_dataset.img_wh
                    select_inds = torch.randperm(int(H * W)).to(self.device)[
                        : self.cfg.optim.batch_size
                    ]
                    all_ray_idxs = select_inds
                    i, j = ids2pixel(W, H, all_ray_idxs)
                    directions = get_ray_directions_lean(
                        i,
                        j,
                        [test_dataset.focal, test_dataset.focal],
                        [test_dataset.cx, test_dataset.cy],
                    )
                    cam2world = self.get_cam2world(
                        global_img_idxs.reshape(self.cfg.optim.num_s_imgs, 1)
                    )
                    cam2hex = (
                        cam2world.clone().detach()
                    )  # do not detach when using gradients for pose optim
                    cam2hex[:, :3, 3] += self.world2hexs[-1]
                    rays_train = get_rays_lean(
                        directions.to(self.device),
                        cam2hex.repeat_interleave(
                            directions.shape[0] // self.cfg.optim.num_s_imgs, dim=0
                        ),
                    ).reshape(-1, 6)

                else:
                    W, H = train_dataset.img_wh
                    data = train_dataset[img_i]
                    rays_train = data["rays"]
                    ray_idxs_repeated = torch.arange(
                        0, self.cfg.optim.num_s_imgs
                    ).repeat_interleave(
                        self.cfg.optim.batch_size // self.cfg.optim.num_s_imgs, dim=0
                    ).to(
                        self.device
                    ) * (
                        W * H
                    )

                    select_inds = torch.randperm(rays_train.shape[1]).to(self.device)[
                        : self.cfg.optim.batch_size
                    ]
                    rays_train = rays_train.view(-1, 6)[
                        (select_inds + ray_idxs_repeated).detach().cpu()
                    ]

                rgb_train, frame_time = (
                    data["rgbs"].to(self.device).view(-1, 3),
                    data["time"],
                )
                spec_mask = data["spec_mask"].to(self.device).view(-1, 1)[select_inds]

                ray_idxs_repeated = torch.arange(
                    0, self.cfg.optim.num_s_imgs
                ).repeat_interleave(
                    self.cfg.optim.batch_size // self.cfg.optim.num_s_imgs, dim=0
                ).to(
                    self.device
                ) * (
                    W * H
                )
                select_inds += ray_idxs_repeated
                select_inds = select_inds.detach().cpu()
                rgb_train = rgb_train[select_inds]

                frame_time = frame_time.view(-1, 1)[select_inds]

                if self.depth_data:
                    train_depth = (
                        data["depths"].to(self.device).view(-1, 1)[select_inds]
                    )
                if self.train_dataset.load_tool_mask:
                    tool_mask = (
                        data["tool_mask"].to(self.device).view(-1, 1)[select_inds]
                    )
                if self.cfg.data.flow_data:
                    gt_fwd = data["fwd"].to(self.device).view(-1, 2)[select_inds]
                    gt_bwd = data["bwd"].to(self.device).view(-1, 2)[select_inds]
                    fwd_mask = data["fwd_mask"].to(self.device).view(-1, 1)[select_inds]
                    bwd_mask = data["bwd_mask"].to(self.device).view(-1, 1)[select_inds]

                ray_idx = select_inds
                img_idxs = img_i

        else:
            raise NotImplementedError(
                "Only image-based sampling is implemented for now."
            )

        return (
            rays_train,
            rgb_train,
            frame_time,
            train_depth,
            gt_fwd,
            gt_bwd,
            fwd_mask,
            bwd_mask,
            ray_idx,
            img_idxs,
            train_test_poses,
            tool_mask,
            spec_mask,
        )
