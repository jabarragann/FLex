import copy
import math
import os
import sys

import cv2
import imageio
import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from pytorch_msssim import ms_ssim as MS_SSIM
from tqdm.auto import tqdm, trange

from flex.dataloader.ray_utils import get_ray_directions_lean, get_rays_lean, ids2pixel
from flex.render.util.metric import rgb_lpips, rgb_ssim
from flex.render.util.util import (
    add_text_to_img,
    draw_poses,
    flow_to_image,
    get_all_poses,
    map_train_bounds2global,
    visualize_depth_numpy,
)
from local.util import eval_poses, icosphere, mtx_to_sixD, read_poses, sixD_to_mtx


def get_cam2world(poses_rot, poses_t, img_idxs=None, initial_pose=None):
    if img_idxs is not None:
        poses_rot = torch.stack([poses_rot[img_idx] for img_idx in img_idxs], dim=0)
        poses_t = torch.stack([poses_t[img_idx] for img_idx in img_idxs], dim=0)
    else:
        poses_rot = torch.stack(list(poses_rot), dim=0)
        poses_t = torch.stack(list(poses_t), dim=0)

    return torch.cat([sixD_to_mtx(poses_rot), poses_t[..., None]], dim=-1)


def OctreeRender_trilinear_fast(
    rays,
    time,
    model,
    chunk=4096,
    N_samples=-1,
    ndc_ray=False,
    white_bg=True,
    is_train=False,
    device="cuda",
):
    """
    Batched rendering function.
    """
    rgbs, alphas, depth_maps, z_vals, xyz_sampled, weights = [], [], [], [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)
        time_chunk = time[chunk_idx * chunk : (chunk_idx + 1) * chunk].to(device)

        rgb_map, depth_map, alpha_map, z_val_map, xyz_sampled_map, weights_map = model(
            rays_chunk,
            time_chunk,
            is_train=is_train,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            N_samples=N_samples,
        )

        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
        alphas.append(alpha_map)
        z_vals.append(z_val_map)
        xyz_sampled.append(xyz_sampled_map)
        weights.append(weights_map)

    return (
        torch.cat(rgbs),
        torch.cat(alphas),
        torch.cat(depth_maps),
        torch.cat(z_vals),
        torch.cat(xyz_sampled),
        torch.cat(weights),
    )


def create_pc(
    dataset,
    model,
    N_samples,
    ndc_ray,
    white_bg,
    device,
    near_far,
    pc_test_idx,
    savePath=None,
    prefix="",
    cfg=None,  # not valid option
):
    W, H = dataset.img_wh
    focal = dataset.focal

    if cfg.compute_pc_cd:
        pc_test_idx = torch.arange(0, len(dataset)).tolist()
        pcd_errors = []

    for idx in pc_test_idx:
        data = dataset[idx]
        samples, gt_rgb, sample_times, gt_depth = (
            data["rays"],
            data["rgbs"],
            data["time"],
            data["depths"],
        )
        # explicitly load gt depth maps:
        rays = samples.view(-1, samples.shape[-1])
        times = sample_times.view(-1, sample_times.shape[-1])

        rgb_map, _, depth_map, _, _, _ = OctreeRender_trilinear_fast(
            rays,
            times,
            model,
            chunk=4096,
            N_samples=N_samples,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )

        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_map, depth_map = (
            rgb_map.reshape(H, W, 3).cpu(),
            depth_map.reshape(H, W).cpu(),
        )
        gt_rgb_map = gt_rgb.view(H, W, 3).cpu()
        gt_depth_map = gt_depth.reshape(H, W).cpu()

        depth_map = np.nan_to_num(depth_map).astype(np.float32)
        gt_depth_map = np.nan_to_num(gt_depth_map).astype(np.float32)
        rgb_map = (rgb_map.numpy() * 255).astype("uint8")
        gt_rgb_map = (gt_rgb_map.numpy() * 255).astype("uint8")

        rgb_im = o3d.geometry.Image(rgb_map)
        depth_im = o3d.geometry.Image(depth_map)
        # gt
        gt_rgb_im = o3d.geometry.Image(gt_rgb_map)
        gt_depth_im = o3d.geometry.Image(gt_depth_map)

        # convert to rgbd map
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_im, depth_im, convert_rgb_to_intensity=False
        )
        gt_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            gt_rgb_im, gt_depth_im, convert_rgb_to_intensity=False
        )
        # create pcd from rgbd map
        # extrinsic = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        extrinsic = dataset.poses[idx]
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            width=W, height=H, fx=focal, fy=focal, cx=dataset.cx, cy=dataset.cy
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsic, extrinsic=extrinsic
        )
        gt_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            gt_rgbd_image, intrinsic, extrinsic=extrinsic
        )

        # save pcd
        if not cfg.compute_pc_cd:
            os.makedirs(f"{savePath}/pc", exist_ok=True)
            o3d.io.write_point_cloud(
                f"{savePath}/pc/{prefix}{idx:03d}.ply", pcd, write_ascii=False
            )
            o3d.io.write_point_cloud(
                f"{savePath}/pc/{prefix}_gt_{idx:03d}.ply", gt_pcd, write_ascii=False
            )

            if cfg.pc_icp:
                result = o3d.pipelines.registration.registration_icp(
                    gt_pcd,
                    pcd,
                    max_correspondence_distance=0.001,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=2000
                    ),
                )
                print(result)
                transformation = result.transformation
                icp_pcd = pcd.transform(transformation)
                o3d.io.write_point_cloud(
                    f"{savePath}/pc/{prefix}_icp_{idx:03d}.ply",
                    icp_pcd,
                    write_ascii=False,
                )

            print("saved Point Clouds for test idx: " + str(idx))

        else:
            depth_error = torch.mean(
                torch.abs(torch.tensor(depth_map) - torch.tensor(gt_depth_map))
            )
            pcd_errors.append(depth_error)
            print(
                f"Computed Depth Error for test idx: {idx}, Error: {depth_error.detach().cpu()}"
            )

    if cfg.compute_pc_cd:
        with open(f"{savePath}/{prefix}pcd_mean.txt", "w") as f:
            f.write(f"CD: {np.mean(pcd_errors)}\n")
            print(f"CD: {np.mean(pcd_errors)}\n")
            for i in range(len(pcd_errors)):
                f.write(f"Index {i}, CD: {pcd_errors[i]}\n")


def create_pc_local(
    dataset,
    N_samples,
    ndc_ray,
    white_bg,
    device,
    pc_test_idx,
    savePath,
    prefix,
    cfg,
    blending_weights,
    poses_rot,
    poses_t,
    world2hexs,
    modelsdirectory,
):
    W, H = dataset.img_wh
    focal = dataset.focal
    near_far = dataset.near_far

    if cfg.compute_pc_cd:
        pc_test_idx = torch.arange(0, len(dataset)).tolist()
        pcd_errors = []

    for idx in pc_test_idx:
        data = dataset[idx]

        ray_idxs = torch.arange(0, W * H)
        i, j = ids2pixel(W, H, ray_idxs)
        # directions = get_ray_directions_lean(i, j, [dataset.focal, dataset.focal], [W/2, H/2])
        directions = get_ray_directions_lean(
            i, j, [dataset.focal, dataset.focal], [dataset.cx, dataset.cy]
        )
        idx_repeat = (torch.ones(directions.shape[0]) * idx * 8).to(torch.long)
        cam2hexs = {}
        if cfg.local.use_preprocessed_poses:
            cam2world = torch.cat(
                [
                    poses_rot[idx_repeat].detach().cpu(),
                    poses_t[idx_repeat][..., None].detach().cpu(),
                ],
                dim=-1,
            )
        else:
            cam2world = get_cam2world(poses_rot, poses_t, idx_repeat)
        active_hex_ids = torch.nonzero(blending_weights[idx * 8, :])[:, 0].tolist()
        for model_id in active_hex_ids:
            cam2hex = cam2world.clone().detach().cpu()
            cam2hex[:, :3, 3] += world2hexs[model_id].detach().cpu()

            cam2hexs[model_id] = cam2hex

        gt_rgb, sample_times, gt_depth = data["rgbs"], data["time"], data["depths"]

        # reshape blending weights
        blending_weights_expanded = blending_weights[idx_repeat]
        first = True

        # init variables
        rgb_map = torch.zeros_like(directions, device=device)
        depth_map = torch.zeros_like(directions[..., 0], device=device)

        for model_id in active_hex_ids:
            blending_weight = blending_weights_expanded[:, model_id]
            cam2hex = cam2hexs[model_id]
            model = torch.load(f"{modelsdirectory}_{model_id}.th", map_location=device)
            samples = get_rays_lean(directions, cam2hex)
            rays = samples.view(-1, samples.shape[-1])
            times = sample_times.view(-1, sample_times.shape[-1])
            (
                rgb_map_t,
                _,
                depth_map_t,
                _,
                xyz_sampled_t,
                weights_t,
            ) = OctreeRender_trilinear_fast(
                rays,
                times,
                model,
                chunk=cfg.optim.batch_size // 2,
                N_samples=N_samples,
                ndc_ray=ndc_ray,
                white_bg=white_bg,
                device=device,
            )
            # adjust according to blending weights
            rgb_map = rgb_map + rgb_map_t * blending_weight[..., None]
            depth_map = depth_map + depth_map_t * blending_weight

        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_map, depth_map = (
            rgb_map.reshape(H, W, 3).cpu(),
            depth_map.reshape(H, W).cpu(),
        )
        gt_rgb_map = gt_rgb.view(H, W, 3).cpu()
        gt_depth_map = gt_depth.reshape(H, W).cpu()

        depth_map = np.nan_to_num(depth_map).astype(np.float32)
        gt_depth_map = np.nan_to_num(gt_depth_map).astype(np.float32)
        rgb_map = (rgb_map.numpy() * 255).astype("uint8")
        gt_rgb_map = (gt_rgb_map.numpy() * 255).astype("uint8")

        rgb_im = o3d.geometry.Image(rgb_map)
        depth_im = o3d.geometry.Image(depth_map)
        # gt
        gt_rgb_im = o3d.geometry.Image(gt_rgb_map)
        gt_depth_im = o3d.geometry.Image(gt_depth_map)

        # convert to rgbd map
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_im, depth_im, convert_rgb_to_intensity=False
        )
        gt_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            gt_rgb_im, gt_depth_im, convert_rgb_to_intensity=False
        )
        # create pcd from rgbd map
        # extrinsic = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
        extrinsic = dataset.poses[idx]

        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            width=W, height=H, fx=focal, fy=focal, cx=dataset.cx, cy=dataset.cy
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsic, extrinsic=extrinsic
        )
        gt_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            gt_rgbd_image, intrinsic, extrinsic=extrinsic
        )

        # save pcd
        if not cfg.compute_pc_cd:
            os.makedirs(f"{savePath}/pc", exist_ok=True)
            o3d.io.write_point_cloud(
                f"{savePath}/pc/{prefix}{idx:03d}.ply", pcd, write_ascii=False
            )
            o3d.io.write_point_cloud(
                f"{savePath}/pc/{prefix}_gt_{idx:03d}.ply", gt_pcd, write_ascii=False
            )

            if cfg.pc_icp:
                result = o3d.pipelines.registration.registration_icp(
                    gt_pcd,
                    pcd,
                    max_correspondence_distance=0.001,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=2000
                    ),
                )
                print(result)
                transformation = result.transformation
                icp_pcd = pcd.transform(transformation)
                o3d.io.write_point_cloud(
                    f"{savePath}/pc/{prefix}_icp_{idx:03d}.ply",
                    icp_pcd,
                    write_ascii=False,
                )

            print("saved Point Clouds for test idx: " + str(idx))

        else:
            # t_d = torch.median(dyn_depth, dim=-1, keepdim=True).values
            # s_d = torch.mean(torch.abs(dyn_depth - t_d), dim=-1, keepdim=True)
            # dyn_depth_norm = (dyn_depth - t_d) / s_d

            # t_gt = torch.median(gt_depth, dim=-1, keepdim=True).values
            # s_gt = torch.mean(torch.abs(gt_depth - t_gt), dim=-1, keepdim=True)
            # gt_depth_norm = (gt_depth - t_gt) / s_gt
            # dyn_depth_norm, gt_depth_norm, (dyn_depth_norm - gt_depth_norm) ** 2
            depth_error = torch.mean(
                torch.abs(torch.tensor(depth_map) - torch.tensor(gt_depth_map))
            )
            pcd_errors.append(depth_error)
            print(
                f"Computed Depth Error for test idx: {idx}, Error: {depth_error.detach().cpu()}"
            )

    if cfg.compute_pc_cd:
        with open(f"{savePath}/{prefix}pcd_mean.txt", "w") as f:
            f.write(f"CD: {np.mean(pcd_errors)}\n")
            print(f"CD: {np.mean(pcd_errors)}\n")
            for i in range(len(pcd_errors)):
                f.write(f"Index {i}, CD: {pcd_errors[i]}\n")


def render_extra(
    dataset,
    model,
    N_samples,
    ndc_ray,
    white_bg,
    device,
    near_far,
    pc_test_idx,
    savePath=None,
    prefix="",
    repeats=8,
    num_scene=0,
    original_pose=False,
    gen_pc=False,
    cfg=None,
):
    W, H = dataset.img_wh
    focal = dataset.focal
    rgb_maps = []
    count = 0
    find_time_again = True
    if cfg.img_limit < len(dataset):
        total_img = cfg.img_limit * repeats + cfg.img_limit
        orig_imgs = cfg.img_limit
    else:
        total_img = math.ceil(
            (len(dataset) * repeats + len(dataset)) / cfg.eval_every_img
        )
        orig_imgs = len(dataset)

    img_idxs = np.arange(
        0, orig_imgs, cfg.eval_every_img
    )  # set specific img idxs from test set to evaluate for

    for idx in img_idxs:
        # data = dataset[num_scene]
        # idx = img_idxs[-1] # only for quick testing
        data = dataset[idx]
        gt_rgb, gt_depth = data["rgbs"], data["depths"]
        if (
            original_pose
        ):  # keep same pose from specified scene also for different timesteps
            pose = dataset.poses[num_scene]
            time = dataset.all_times[idx]
        else:  # take actual timestep pose
            pose = dataset.poses[idx]
            time = dataset.all_times[idx]

        all_samples, all_sample_times = dataset.get_new_pose_rays(pose, repeats, time)

        for i in range(len(all_samples)):
            samples = all_samples[i]
            sample_times = all_sample_times[i]
            sample_times = sample_times.repeat((len(samples), 1))
            rays = samples.view(-1, samples.shape[-1])
            if find_time_again:
                times = sample_times.view(-1, sample_times.shape[-1])
            (
                rgb_map,
                _,
                depth_map,
                _,
                _,
                _,
            ) = OctreeRender_trilinear_fast(
                rays,
                times,
                model,
                chunk=4096,
                N_samples=N_samples,
                ndc_ray=ndc_ray,
                white_bg=white_bg,
                device=device,
            )
            rgb_map = rgb_map.clamp(0.0, 1.0)
            rgb_map, depth_map = (
                rgb_map.reshape(H, W, 3).cpu(),
                depth_map.reshape(H, W).cpu(),
            )
            gt_rgb_map = gt_rgb.view(H, W, 3).cpu()
            gt_depth_map = gt_depth.reshape(H, W).cpu()

            depth_map = np.nan_to_num(depth_map).astype(np.float32)
            gt_depth_map = np.nan_to_num(gt_depth_map).astype(np.float32)
            rgb_map = (rgb_map.numpy() * 255).astype("uint8")
            gt_rgb_map = (gt_rgb_map.numpy() * 255).astype("uint8")

            rgb_maps.append(rgb_map)
            count += 1
            print("rendered: " + str(count) + "/" + str(total_img))

            if gen_pc:
                rgb_im = o3d.geometry.Image(rgb_map)
                depth_im = o3d.geometry.Image(depth_map)

                # convert to rgbd map
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    rgb_im, depth_im, convert_rgb_to_intensity=False
                )
                # create pcd from rgbd map
                extrinsic = np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                intrinsic = o3d.camera.PinholeCameraIntrinsic()
                intrinsic.set_intrinsics(
                    width=W, height=H, fx=focal, fy=focal, cx=dataset.cx, cy=dataset.cy
                )
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                    rgbd_image, intrinsic, extrinsic=extrinsic
                )
                # save pcd
                if count == 1:
                    savePath = (
                        os.path.split(os.path.split(savePath)[0])[0]
                        + "/render_extra_poses"
                    )
                    os.makedirs(f"{savePath}/pc", exist_ok=True)
                o3d.io.write_point_cloud(
                    f"{savePath}/pc/{prefix}{idx:03d}.ply", pcd, write_ascii=False
                )
                print("saved Point Clouds for test idx: " + str(idx))

    if cfg.eval_every_img > 1:
        fps = 10 / cfg.eval_every_img
        if fps < 1:
            fps = 1
    elif total_img > 150:
        fps = 25
    else:
        fps = 10

    if not gen_pc:
        savePath = os.path.split(os.path.split(savePath)[0])[0] + "/render_extra_poses"
    os.makedirs(savePath, exist_ok=True)
    imageio.mimwrite(
        f"{savePath}/{prefix}video.mp4",
        np.stack(rgb_maps),
        fps=fps,
        format="FFMPEG",
        quality=8,
    )


def render_extra_local(
    dataset,
    N_samples,
    ndc_ray,
    white_bg,
    device,
    near_far,
    pc_test_idx,
    savePath=None,
    prefix="",
    repeats=8,
    num_scene=0,
    original_pose=False,
    gen_pc=False,
    cfg=None,
    blending_weights=None,
    poses_rot=None,
    poses_t=None,
    world2hexs=None,
    modelsdirectory=None,
):
    W, H = dataset.img_wh
    focal = dataset.focal
    rgb_maps = []
    count = 0
    find_time_again = True
    every_extra_frame = 25
    if cfg.img_limit < len(dataset):
        total_img = cfg.img_limit * repeats + cfg.img_limit
        orig_imgs = cfg.img_limit
    else:
        total_img = math.ceil(
            (len(dataset) * repeats + len(dataset)) / cfg.eval_every_img
        )
        orig_imgs = len(dataset)

    img_idxs = np.arange(
        0, orig_imgs, cfg.eval_every_img
    )  # set specific img idxs from test set to evaluate for

    # img_idxs = np.arange(0, len(poses_rot)+12)
    img_idxs = np.arange(0, len(dataset) + 1)
    total_img = len(img_idxs) + ((((len(img_idxs) - 1) // every_extra_frame) + 1) * 11)
    icosphere_vertices, _ = icosphere(1)

    repeat_count = 0
    idx = 0
    while idx < img_idxs[-1]:
        if repeat_count == 0 and idx > len(dataset):
            break
        elif repeat_count > 0:
            idx -= 1

        data = dataset[idx]
        gt_rgb, gt_depth = data["rgbs"], data["depths"]

        ray_idxs = torch.arange(0, H * W)  # only needed when computing for entire image
        i, j = ids2pixel(W, H, ray_idxs)
        # directions = get_ray_directions_lean(i, j, [dataset.focal, dataset.focal], [W/2, H/2])
        directions = get_ray_directions_lean(
            i, j, [dataset.focal, dataset.focal], [dataset.cx, dataset.cy]
        )
        idx_repeat = (torch.ones(directions.shape[0]) * idx * 8).to(torch.long)
        cam2hexs = {}
        if cfg.local.use_preprocessed_poses:
            cam2world = torch.cat(
                [
                    poses_rot[idx_repeat].detach().cpu(),
                    poses_t[idx_repeat][..., None].detach().cpu(),
                ],
                dim=-1,
            )
        else:
            cam2world = get_cam2world(poses_rot, poses_t, idx_repeat)
        active_hex_ids = torch.nonzero(blending_weights[idx * 8, :])[:, 0].tolist()

        if (idx) % every_extra_frame == 0:
            # do sth to pose
            t_range = 0.1  # 0.05
            t_noises = icosphere_vertices * t_range
            t = cam2world[:, :3, 3]
            cam2world[:, :3, 3] = t + torch.tensor(t_noises[repeat_count])
            repeat_count += 1
            if repeat_count == 12:
                repeat_count = 0

        for model_id in active_hex_ids:
            cam2hex = cam2world.clone().detach().cpu()
            cam2hex[:, :3, 3] += world2hexs[model_id].detach().cpu()

            cam2hexs[model_id] = cam2hex

        sample_times = dataset.all_times[idx]

        # reshape blending weights
        blending_weights_expanded = blending_weights[idx_repeat]
        first = True

        # init variables
        rgb_map = torch.zeros_like(directions, device=device)
        depth_map = torch.zeros_like(directions[..., 0], device=device)

        for model_id in active_hex_ids:
            blending_weight = blending_weights_expanded[:, model_id]
            cam2hex = cam2hexs[model_id]
            model = torch.load(f"{modelsdirectory}_{model_id}.th", map_location=device)
            samples = get_rays_lean(directions, cam2hex)
            rays = samples.view(-1, samples.shape[-1])
            times = sample_times.view(-1, sample_times.shape[-1])

            (
                rgb_map_t,
                _,
                depth_map_t,
                _,
                xyz_sampled_t,
                weights_t,
            ) = OctreeRender_trilinear_fast(
                rays,
                times,
                model,
                chunk=cfg.optim.batch_size,  # //2,
                N_samples=N_samples,
                ndc_ray=ndc_ray,
                white_bg=white_bg,
                device=device,
            )
            # adjust according to blending weights
            rgb_map = rgb_map + rgb_map_t * blending_weight[..., None]
            depth_map = depth_map + depth_map_t * blending_weight
            if cfg.data.flow_data and cfg.model.predict_flow:
                blending_weight_3d = blending_weight[..., None].repeat(
                    1, xyz_sampled_t.shape[1]
                )
                if first:
                    if not cfg.model.opt_from_depth:
                        scene_flow = scene_flow_t * blending_weight_3d[..., None]
                    xyz_sampled = xyz_sampled_t * blending_weight_3d[..., None]
                    weights = weights_t * blending_weight_3d
                    first = False
                else:
                    if not cfg.model.opt_from_depth:
                        scene_flow = (
                            scene_flow + scene_flow_t * blending_weight_3d[..., None]
                        )
                    xyz_sampled = (
                        xyz_sampled + xyz_sampled_t * blending_weight_3d[..., None]
                    )
                    weights = weights + weights_t * blending_weight_3d

        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_map, depth_map = (
            rgb_map.reshape(H, W, 3).cpu(),
            depth_map.reshape(H, W).cpu(),
        )
        gt_rgb_map = gt_rgb.view(H, W, 3).cpu()
        gt_depth_map = gt_depth.reshape(H, W).cpu()

        depth_map = np.nan_to_num(depth_map).astype(np.float32)
        gt_depth_map = np.nan_to_num(gt_depth_map).astype(np.float32)
        rgb_map = (rgb_map.numpy() * 255).astype("uint8")
        gt_rgb_map = (gt_rgb_map.numpy() * 255).astype("uint8")

        rgb_maps.append(rgb_map)
        count += 1
        print("rendered: " + str(count) + "/" + str(total_img))
        idx += 1

        if gen_pc:
            rgb_im = o3d.geometry.Image(rgb_map)
            depth_im = o3d.geometry.Image(depth_map)

            # convert to rgbd map
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_im, depth_im, convert_rgb_to_intensity=False
            )
            # create pcd from rgbd map
            extrinsic = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            intrinsic = o3d.camera.PinholeCameraIntrinsic()
            intrinsic.set_intrinsics(
                width=W, height=H, fx=focal, fy=focal, cx=dataset.cx, cy=dataset.cy
            )
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, intrinsic, extrinsic=extrinsic
            )
            # save pcd
            if count == 1:
                savePath = (
                    os.path.split(os.path.split(savePath)[0])[0] + "/render_extra_poses"
                )
                os.makedirs(f"{savePath}/pc", exist_ok=True)
            o3d.io.write_point_cloud(
                f"{savePath}/pc/{prefix}{idx:03d}.ply", pcd, write_ascii=False
            )
            print("saved Point Clouds for test idx: " + str(idx))

    if total_img > 200:
        fps = 20
    else:
        fps = 10

    if not gen_pc:
        savePath = os.path.split(os.path.split(savePath)[0])[0] + "/render_extra_poses"
    os.makedirs(savePath, exist_ok=True)
    imageio.mimwrite(
        f"{savePath}/{prefix}video.mp4",
        np.stack(rgb_maps),
        fps=fps,
        format="FFMPEG",
        quality=8,
    )


def render_new_poses(
    dataset,
    N_samples,
    ndc_ray,
    white_bg,
    device,
    near_far,
    new_pose_file,
    poses_rot,
    poses_t,
    savePath=None,
    prefix="",
    original_pose=False,
    gen_pc=False,
    cfg=None,
    blending_weights=None,
    world2hexs=None,
    modelsdirectory=None,
):
    W, H = dataset.img_wh
    focal = dataset.focal
    rgb_maps = []
    count = 0
    # preprocess poses
    poses, pose_stamps = read_poses(pose_path=new_pose_file, return_stamps=True)
    initial_pose = dataset.poses[0]
    # pose stamps need to be in img idxs equivalent to train and test frames!
    assert pose_stamps.all() < len(poses_rot)
    # flip from OpenCV to OpenGL
    flip = np.eye(4)
    flip[1, 1] = -1
    flip[2, 2] = -1
    poses = np.matmul(poses, flip)
    # readjust poses to identical scale and shift than trained poses
    if cfg.local.use_preprocessed_poses:
        poses[..., 3] = poses[..., 3] - (dataset.trans_center_shift[:3] + [0])
    poses[:, :3, 3] = (poses[:, :3, 3]) / dataset.depth_scale_factor

    total_img = len(poses)

    savePath = os.path.split(os.path.split(savePath)[0])[0] + "/render_new_poses"
    os.makedirs(f"{savePath}", exist_ok=True)

    for idx in pose_stamps:
        idx = int(idx)
        ray_idxs = torch.arange(0, H * W)  # only needed when computing for entire image
        i, j = ids2pixel(W, H, ray_idxs)
        directions = get_ray_directions_lean(
            i, j, [dataset.focal, dataset.focal], [dataset.cx, dataset.cy]
        )
        idx_repeat = (torch.ones(directions.shape[0]) * idx).to(torch.long)
        cam2hexs = {}
        if cfg.local.use_preprocessed_poses:
            # fix position
            if idx == 0:
                pose = np.eye(4, dtype=np.float32)
            else:
                pose = np.linalg.inv(initial_pose) @ poses[idx]

            cam2world = torch.tensor(pose.reshape(1, 4, 4)[idx_repeat * 0])[:, :3, :4]
        else:
            cam2world = torch.tensor((poses[count]).reshape(1, 4, 4)[idx_repeat * 0])[
                :, :3, :4
            ]
            c2w = get_cam2world(poses_rot.detach().cpu(), poses_t.detach().cpu())
            cam2world = c2w[idx_repeat][:, :3, :4]
        active_hex_ids = torch.nonzero(blending_weights[idx, :])[:, 0].tolist()

        for model_id in active_hex_ids:
            cam2hex = cam2world.clone().detach().cpu().float()
            cam2hex[:, :3, 3] += world2hexs[model_id].detach().cpu()

            cam2hexs[model_id] = cam2hex

        sample_times = torch.tensor((idx / (len(poses_rot) - 1)) * 2.0 - 1.0).expand(
            ray_idxs.shape[0], 1
        )

        # reshape blending weights
        blending_weights_expanded = blending_weights[idx_repeat]
        first = True

        # init variables
        rgb_map = torch.zeros_like(directions, device=device)
        depth_map = torch.zeros_like(directions[..., 0], device=device)

        for model_id in active_hex_ids:
            blending_weight = blending_weights_expanded[:, model_id]
            cam2hex = cam2hexs[model_id]
            model = torch.load(f"{modelsdirectory}_{model_id}.th", map_location=device)
            samples = get_rays_lean(directions, cam2hex)
            rays = samples.view(-1, samples.shape[-1])
            times = sample_times.view(-1, sample_times.shape[-1])

            (
                rgb_map_t,
                _,
                depth_map_t,
                _,
                xyz_sampled_t,
                weights_t,
            ) = OctreeRender_trilinear_fast(
                rays,
                times,
                model,
                chunk=cfg.optim.batch_size,
                N_samples=N_samples,
                ndc_ray=ndc_ray,
                white_bg=white_bg,
                device=device,
            )
            # adjust according to blending weights
            rgb_map = rgb_map + rgb_map_t * blending_weight[..., None]
            depth_map = depth_map + depth_map_t * blending_weight
            if cfg.data.flow_data and cfg.model.predict_flow:
                blending_weight_3d = blending_weight[..., None].repeat(
                    1, xyz_sampled_t.shape[1]
                )
                if first:
                    xyz_sampled = xyz_sampled_t * blending_weight_3d[..., None]
                    weights = weights_t * blending_weight_3d
                    first = False
                else:
                    xyz_sampled = (
                        xyz_sampled + xyz_sampled_t * blending_weight_3d[..., None]
                    )
                    weights = weights + weights_t * blending_weight_3d

        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_map, depth_map = (
            rgb_map.reshape(H, W, 3).cpu(),
            depth_map.reshape(H, W).cpu(),
        )

        depth_map = np.nan_to_num(depth_map).astype(np.float32)
        rgb_map = (rgb_map.numpy() * 255).astype("uint8")

        # save images
        imageio.imwrite(f"{savePath}/{prefix}{idx:03d}.png", rgb_map)

        rgb_maps.append(rgb_map)
        count += 1
        print("rendered: " + str(count) + "/" + str(total_img))

        if gen_pc:
            rgb_im = o3d.geometry.Image(rgb_map)
            depth_im = o3d.geometry.Image(depth_map)

            # convert to rgbd map
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_im, depth_im, convert_rgb_to_intensity=False
            )
            # create pcd from rgbd map
            extrinsic = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            intrinsic = o3d.camera.PinholeCameraIntrinsic()
            intrinsic.set_intrinsics(
                width=W, height=H, fx=focal, fy=focal, cx=dataset.cx, cy=dataset.cy
            )
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, intrinsic, extrinsic=extrinsic
            )
            # save pcd
            if count == 1:
                os.makedirs(f"{savePath}/pc", exist_ok=True)
            o3d.io.write_point_cloud(
                f"{savePath}/pc/{prefix}{idx:03d}.ply", pcd, write_ascii=False
            )
            print("saved Point Clouds for test idx: " + str(idx))

    if total_img > 200:
        fps = 20
    else:
        fps = 10

    os.makedirs(savePath, exist_ok=True)
    imageio.mimwrite(
        f"{savePath}/{prefix}video.mp4",
        np.stack(rgb_maps),
        fps=fps,
        format="FFMPEG",
        quality=8,
    )


def test_weights(
    dataset,
    model,
    N_samples,
    ndc_ray,
    white_bg,
    device,
    near_far,
    savePath,
    cfg,
):
    """
    Test and plot model density weights for specific rays across several time steps
    """
    W, H = dataset.img_wh
    focal = dataset.focal
    # idx = 10
    idx = cfg.test_view
    repeats = 11
    start_shift = -int(repeats / 2)
    time_step_diff = 2 / 1000
    count = 0
    all_weights = []
    ray_idxs = cfg.ray_idxs
    # N_samples = 200 # reduced samples for increased plot visibility

    for i in range(repeats):
        # data = dataset[num_scene]
        # idx = img_idxs[-1] # only for quick testing
        data = dataset[idx]
        gt_rgb, gt_depth = data["rgbs"], data["depths"]
        pose = dataset.poses[idx]
        time = dataset.all_times[idx]
        time = time + ((i + start_shift) * time_step_diff)
        all_samples = dataset.all_rays[idx]

        samples = all_samples[ray_idxs]
        sample_times = time
        # sample_times = sample_times.repeat((len(samples),1))
        sample_times = sample_times[ray_idxs]
        rays = samples.view(-1, samples.shape[-1])
        rgb_map, _, depth_map, _, _, weights = OctreeRender_trilinear_fast(
            rays,
            sample_times,
            model,
            chunk=4096,
            N_samples=N_samples,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )
        all_weights.append(weights.detach().cpu().numpy())
        count += 1
        print("rendered: " + str(count) + "/" + str(repeats))

    savePath = os.path.split(os.path.split(savePath)[0])[0] + "/test_weights"
    os.makedirs(f"{savePath}", exist_ok=True)
    t = np.arange(0, all_weights[0].shape[-1])
    # ax.plot(t, all_weights[0], 'r--', t, all_weights[1], 'bs', t, all_weights[2], 'g^')
    for j in range(all_weights[0].shape[0]):
        fig, ax = plt.subplots()
        for weights in all_weights:
            ax.plot(t, weights[j])
        var = np.stack(all_weights, 0)
        var = np.var(var[:, j])
        ax.set_title(
            f"ray at test view {idx} and position: Width:{ray_idxs[j] % W} and Height:{(ray_idxs[j] // W) % H} Var:{var}"
        )
        fig.savefig(
            f"{savePath}/weight_plot_{idx}_{ray_idxs[j] % W}_{(ray_idxs[j] // W) % H}.png"
        )
        pixel_rgb = copy.deepcopy(gt_rgb.detach().cpu())
        pixel_rgb = pixel_rgb.clamp(0.0, 1.0).detach().cpu()
        pixel_rgb = (pixel_rgb.numpy() * 255).astype("uint8")
        pixel_rgb[(ray_idxs[j] // W) % H, ray_idxs[j] % W] = np.array([0, 255, 0])
        pixel_mask = torch.ones((H, W, 3))
        pixel_mask[(ray_idxs[j] // W) % H, ray_idxs[j] % W] = torch.tensor([0, 255, 0])
        imageio.imwrite(
            f"{savePath}/img_{idx}_{ray_idxs[j] % W}_{(ray_idxs[j] // W) % H}.png",
            pixel_rgb,
        )
        imageio.imwrite(
            f"{savePath}/img_mask_{idx}_{ray_idxs[j] % W}_{(ray_idxs[j] // W) % H}.png",
            pixel_mask,
        )


def test_weights_local(
    dataset,
    N_samples,
    ndc_ray,
    white_bg,
    device,
    near_far,
    savePath,
    cfg,
    blending_weights,
    poses_rot,
    poses_t,
    world2hexs,
    modelsdirectory,
):
    """
    Test and plot model density weights for specific rays across several time steps
    """
    W, H = dataset.img_wh
    focal = dataset.focal
    idx = cfg.test_view
    repeats = 1
    start_shift = 0
    time_step_diff = 2 / 1000
    count = 0
    all_weights = []
    ray_idxs = cfg.ray_idxs
    ray_idxs = torch.tensor(ray_idxs)

    for i in range(repeats):
        # data = dataset[num_scene]
        # idx = img_idxs[-1] # only for quick testing
        data = dataset[idx]
        gt_rgb, gt_depth, fwd_flow, bwd_flow = (
            data["rgbs"],
            data["depths"],
            data["fwd"],
            data["bwd"],
        )
        pose = dataset.poses[idx]
        time = dataset.all_times[idx]
        time = time + ((i + start_shift) * time_step_diff)

        ray_idxs = torch.arange(0, H * W)  # only needed when computing for entire image
        i, j = ids2pixel(W, H, ray_idxs)
        # directions = get_ray_directions_lean(i, j, [dataset.focal, dataset.focal], [W/2, H/2])
        directions = get_ray_directions_lean(
            i, j, [dataset.focal, dataset.focal], [dataset.cx, dataset.cy]
        )
        idx_repeat = (torch.ones(directions.shape[0]) * idx * 8).to(torch.long)
        cam2hexs = {}
        if cfg.local.use_preprocessed_poses:
            cam2world = torch.cat(
                [
                    poses_rot[idx_repeat].detach().cpu(),
                    poses_t[idx_repeat][..., None].detach().cpu(),
                ],
                dim=-1,
            )
        else:
            cam2world = get_cam2world(poses_rot, poses_t, idx_repeat)
        active_hex_ids = torch.nonzero(blending_weights[idx * 8, :])[:, 0].tolist()
        for model_id in active_hex_ids:
            cam2hex = cam2world.clone().detach().cpu()
            cam2hex[:, :3, 3] += world2hexs[model_id].detach().cpu()

            cam2hexs[model_id] = cam2hex

        sample_times = time
        sample_times = sample_times[ray_idxs]

        # reshape blending weights
        blending_weights_expanded = blending_weights[idx_repeat]
        first = True

        # init variables
        rgb_map = torch.zeros_like(directions, device=device)
        depth_map = torch.zeros_like(directions[..., 0], device=device)

        for model_id in active_hex_ids:
            blending_weight = blending_weights_expanded[:, model_id]
            cam2hex = cam2hexs[model_id]
            model = torch.load(f"{modelsdirectory}_{model_id}.th", map_location=device)
            samples = get_rays_lean(directions, cam2hex)
            rays = samples.view(-1, samples.shape[-1])
            times = sample_times.view(-1, sample_times.shape[-1])

            (
                rgb_map_t,
                _,
                depth_map_t,
                _,
                xyz_sampled_t,
                weights_t,
            ) = OctreeRender_trilinear_fast(
                rays,
                times,
                model,
                chunk=cfg.optim.batch_size,  # //2,
                N_samples=N_samples,
                ndc_ray=ndc_ray,
                white_bg=white_bg,
                device=device,
            )
            # adjust according to blending weights
            rgb_map = rgb_map + rgb_map_t * blending_weight[..., None]
            depth_map = depth_map + depth_map_t * blending_weight
            if cfg.data.flow_data and cfg.model.predict_flow:
                blending_weight_3d = blending_weight[..., None].repeat(
                    1, xyz_sampled_t.shape[1]
                )
                if first:
                    if not cfg.model.opt_from_depth:
                        scene_flow = scene_flow_t * blending_weight_3d[..., None]
                    xyz_sampled = xyz_sampled_t * blending_weight_3d[..., None]
                    weights = weights_t * blending_weight_3d
                    first = False
                else:
                    if not cfg.model.opt_from_depth:
                        scene_flow = (
                            scene_flow + scene_flow_t * blending_weight_3d[..., None]
                        )
                    xyz_sampled = (
                        xyz_sampled + xyz_sampled_t * blending_weight_3d[..., None]
                    )
                    weights = weights + weights_t * blending_weight_3d

        all_weights.append(weights.detach().cpu().numpy())
        count += 1
        print("rendered: " + str(count) + "/" + str(repeats))

    savePath = (
        os.path.split(os.path.split(savePath)[0])[0]
        + "/test_weights/"
        + str(cfg.test_view)
    )
    os.makedirs(f"{savePath}", exist_ok=True)

    threshold = 0.004
    mask = torch.var(weights, dim=1)
    print(f"Max Value: {torch.max(mask)}")
    mask = (mask > threshold).view(H, W, 1)
    imageio.imwrite(
        f"{savePath}/thres_mask_{idx}_{threshold}.png",
        mask.long().detach().cpu().numpy(),
    )
    pixel_rgb = copy.deepcopy(gt_rgb.detach().cpu())
    pixel_rgb = pixel_rgb.clamp(0.0, 1.0).detach().cpu()
    pixel_rgb = (pixel_rgb.numpy() * 255).astype("uint8")
    imageio.imwrite(f"{savePath}/img.png", pixel_rgb)
    print(
        f"Fwd Mean: {torch.mean(torch.abs(fwd_flow).cuda().view(-1, 2)[mask.view(-1)], dim=0)} vs {torch.mean(torch.abs(fwd_flow).cuda().view(-1, 2)[~mask.view(-1)], dim=0)}"
    )
    print(
        f"Fwd Min: {torch.min(torch.abs(fwd_flow).view(-1, 2).cuda()[mask.view(-1)], dim=0)[0]} vs {torch.min(torch.abs(fwd_flow).cuda().view(-1, 2)[~mask.view(-1)], dim=0)[0]}"
    )
    print(
        f"Fwd Max: {torch.max(torch.abs(fwd_flow).view(-1, 2).cuda()[mask.view(-1)], dim=0)[0]} vs {torch.max(torch.abs(fwd_flow).cuda().view(-1, 2)[~mask.view(-1)], dim=0)[0]}"
    )
    fwd_flow_magn = torch.abs(fwd_flow).view(-1, 2)
    fwd_flow_magn = torch.linalg.norm(fwd_flow_magn, dim=1)
    fwd_max = torch.max(fwd_flow_magn)
    fwd_flow_magn = fwd_flow_magn / fwd_max
    fwd_flow_magn = fwd_flow_magn.view(H, W, 1).clamp(0.0, 1.0)
    fwd_flow_magn = (fwd_flow_magn.numpy() * 255).astype("uint8")
    imageio.imwrite(f"{savePath}/fwd_magn.png", fwd_flow_magn)

    bwd_flow_magn = torch.abs(bwd_flow).view(-1, 2)
    bwd_flow_magn = torch.linalg.norm(bwd_flow_magn, dim=1)
    bwd_max = torch.max(bwd_flow_magn)
    bwd_flow_magn = bwd_flow_magn / bwd_max
    bwd_flow_magn = bwd_flow_magn.view(H, W, 1).clamp(0.0, 1.0)
    bwd_flow_magn = (bwd_flow_magn.numpy() * 255).astype("uint8")
    imageio.imwrite(f"{savePath}/bwd_magn.png", bwd_flow_magn)

    mask = mask.repeat(1, 1, 3)
    fwd_flow_magn = fwd_flow_magn.repeat(3, axis=-1)
    bwd_flow_magn = bwd_flow_magn.repeat(3, axis=-1)

    out_show = np.hstack(
        [
            add_text_to_img(pixel_rgb, "rgb"),
            add_text_to_img(
                mask.long().detach().cpu().numpy(), "var. mask: " + str(threshold)
            ),
            add_text_to_img(fwd_flow_magn.astype("uint8"), "fwd magnitude"),
            add_text_to_img(bwd_flow_magn.astype("uint8"), "bwd magnitude"),
        ]
    )
    imageio.imwrite(f"{savePath}/all_var_{idx}_thres_{threshold}.png", out_show)


@torch.no_grad()
def evaluation(
    test_dataset,
    model,
    cfg,
    savePath=None,
    N_vis=5,
    prefix="",
    N_samples=-1,
    white_bg=False,
    ndc_ray=False,
    compute_extra_metrics=True,
    device="cuda",
    image_bound=None,
    world2hexs=None,
    num_train_images=None,
    poses_rot=None,
    poses_t=None,
    blending_weights=None,
    hexplanes=None,
    modelsdirectory=None,
):
    """
    Evaluate the model on the test rays and compute metrics.
    """
    PSNRs, rgb_maps, depth_maps, gt_depth_maps, gt_rgb_maps, poses_vis = [], [], [], [], [], []  # fmt: skip

    msssims, ssims, l_alex, l_vgg, depth_errors = [], [], [], [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    if cfg.local_models:
        # assume every 8th frame belongs to test set from inital sequence # TODO: find more generic based on bbox method for finding corresponding test frames
        # find original test and train view idxs
        test_num = np.arange(len(test_dataset))
        orig_test_num = test_num * 8
        train_images = np.arange(num_train_images)
        for i in range(len(train_images)):
            for test_img in orig_test_num:
                if train_images[i] >= test_img:
                    train_images[i] += 1
                else:
                    break
        adj_image_bound = train_images[image_bound[0] : image_bound[1]]
        adj_image_bound = [adj_image_bound[0], adj_image_bound[-1]]
        if image_bound[0] == 0:
            adj_image_bound[0] = 0
        fitting_test_views = []
        for test_img in orig_test_num:
            if test_img >= adj_image_bound[0] and test_img <= adj_image_bound[1]:
                fitting_test_views.append(test_img / 8)

        img_eval_interval = 1 if N_vis < 0 else max(len(fitting_test_views) // N_vis, 1)
        idxs = list(range(0, len(fitting_test_views), img_eval_interval))
        true_idxs = []
        for idx in idxs:
            true_idxs.append(int(fitting_test_views[idx]))
        idxs = true_idxs

    else:
        img_eval_interval = 1 if N_vis < 0 else max(len(test_dataset) // N_vis, 1)
        idxs = list(range(0, len(test_dataset), img_eval_interval))

    if cfg.visualize_4d_vol:
        visualize_4d_vol(test_dataset, cfg, N_samples, model, device, savePath)
        sys.exit()

    # generate pc's only for predefined set of test views
    if cfg.create_pc and not cfg.render_extra:
        create_pc(
            test_dataset,
            model,
            N_samples,
            ndc_ray,
            white_bg,
            device,
            near_far,
            cfg.pc_test_idx,
            savePath,
            prefix,
            cfg,
        )
        sys.exit()

    # generate entirely new views and possibly also pc's
    if cfg.render_extra:
        render_extra(
            test_dataset,
            model,
            N_samples,
            ndc_ray,
            white_bg,
            device,
            near_far,
            cfg.pc_test_idx,
            savePath,
            prefix,
            repeats=cfg.render_extra_repeats,
            num_scene=cfg.num_scene,
            original_pose=cfg.original_pose,
            gen_pc=cfg.create_pc,
            cfg=cfg,
        )
        sys.exit()

    if cfg.test_weights:
        test_weights(
            test_dataset,
            model,
            N_samples,
            ndc_ray,
            white_bg,
            device,
            near_far,
            savePath,
            cfg,
        )
        sys.exit()

    fwd_flows = []
    bwd_flows = []
    gt_fwd_flows = []
    gt_bwd_flows = []

    for idx in tqdm(idxs):
        data = test_dataset[idx]
        if cfg.local_models and cfg.local.progressive_opt:
            W, H = test_dataset.img_wh
            ray_idx = torch.arange(0, int(W * H))
            i, j = ids2pixel(W, H, ray_idx)
            directions = get_ray_directions_lean(
                i,
                j,
                [test_dataset.focal, test_dataset.focal],
                [test_dataset.cx, test_dataset.cy],
            )
            if type(poses_rot) == list:
                poses_rot = torch.stack(poses_rot, 0)
                poses_t = torch.stack(poses_t, 0)
            idx_repeat = (torch.ones(directions.shape[0]) * idx * 8).to(torch.long)
            cam2world = torch.cat(
                [
                    poses_rot[idx_repeat].detach().cpu(),
                    poses_t[idx_repeat][..., None].detach().cpu(),
                ],
                dim=-1,
            )
            # setup local models
            cam2hexs = {}
            active_hex_ids = torch.nonzero(blending_weights[idx * 8, :])[:, 0].tolist()
            for model_id in active_hex_ids:
                cam2hex = cam2world.clone()
                cam2hex[:, :3, 3] += world2hexs[model_id].detach().cpu()

                cam2hexs[model_id] = cam2hex

        else:
            samples = data["rays"]

        gt_rgb, sample_times = data["rgbs"], data["time"]
        depth = None

        W, H = test_dataset.img_wh

        if cfg.local_models and cfg.local.progressive_opt:
            if (
                cfg.optim.optimize_poses
                and idx == idxs[0]
                and os.path.exists(os.path.join(cfg.data.datadir, "groundtruth.txt"))
            ):
                # os.makedirs(savePath + "/poses", exist_ok=True)
                c2w = get_cam2world(poses_rot.detach().cpu(), poses_t.detach().cpu())

                test_idxs = np.arange(0, len(c2w))
                (
                    ate_rmse,
                    rpe_trans_mean,
                    rpe_rot_mean,
                    trans_error,
                    rpe_trans,
                    rpe_rot,
                ) = eval_poses(
                    data_dir=cfg.data.datadir,
                    depth_scale=test_dataset.depth_scale_factor,
                    test_idxs=test_idxs,
                    pred_list=c2w,
                    delta=1,
                    offset=0,
                    savePath=savePath,
                )

                # save metrics!
                with open(f"{savePath}/{prefix}_pose_mean.txt", "w") as f:
                    f.write(
                        f"ATE-RMSE: {ate_rmse}, Trans-Error: {trans_error.mean()}, RPE-Trans: {rpe_trans_mean}, RPE-Rot: {rpe_rot_mean}\n"
                    )
                    print(f"Trajectory errors until idx: {len(c2w)}")
                    print(
                        f"ATE-RMSE: {ate_rmse}, Trans-Error: {trans_error.mean()}, RPE-Trans: {rpe_trans_mean}, RPE-Rot: {rpe_rot_mean}"
                    )

            # init variables
            rgb_map = torch.zeros_like(directions, device=device)
            depth_map = torch.zeros_like(directions[..., 0], device=device)

            # reshape blending weights
            blending_weights_expanded = blending_weights[idx_repeat]
            first = True
            for model_id in active_hex_ids:
                blending_weight = blending_weights_expanded[:, model_id]
                cam2hex = cam2hexs[model_id]
                model = torch.load(
                    f"{modelsdirectory}_{model_id}.th", map_location=device
                )
                samples = get_rays_lean(directions, cam2hex)
                rays = samples.view(-1, samples.shape[-1])
                times = sample_times.view(-1, sample_times.shape[-1])
                (
                    rgb_map_t,
                    _,
                    depth_map_t,
                    _,
                    xyz_sampled_t,
                    weights_t,
                ) = OctreeRender_trilinear_fast(
                    rays,
                    times,
                    model,
                    chunk=cfg.optim.batch_size // 2,
                    N_samples=N_samples,
                    ndc_ray=ndc_ray,
                    white_bg=white_bg,
                    device=device,
                )
                # adjust according to blending weights
                rgb_map = rgb_map + rgb_map_t * blending_weight[..., None]
                depth_map = depth_map + depth_map_t * blending_weight
                if cfg.data.flow_data and cfg.model.predict_flow:
                    blending_weight_3d = blending_weight[..., None].repeat(
                        1, xyz_sampled_t.shape[1]
                    )
                    if first:
                        if not cfg.model.opt_from_depth:
                            scene_flow = scene_flow_t * blending_weight_3d[..., None]
                        xyz_sampled = xyz_sampled_t * blending_weight_3d[..., None]
                        weights = weights_t * blending_weight_3d
                        first = False
                    else:
                        if not cfg.model.opt_from_depth:
                            scene_flow = (
                                scene_flow
                                + scene_flow_t * blending_weight_3d[..., None]
                            )
                        xyz_sampled = (
                            xyz_sampled + xyz_sampled_t * blending_weight_3d[..., None]
                        )
                        weights = weights + weights_t * blending_weight_3d

            torch.cuda.empty_cache()

        else:
            rays = samples.view(-1, samples.shape[-1])
            times = sample_times.view(-1, sample_times.shape[-1])
            (
                rgb_map,
                _,
                depth_map,
                _,
                xyz_sampled,
                weights,
            ) = OctreeRender_trilinear_fast(
                rays,
                times,
                model,
                chunk=cfg.optim.batch_size // 2,
                N_samples=N_samples,
                ndc_ray=ndc_ray,
                white_bg=white_bg,
                device=device,
            )

        rgb_map = rgb_map.clamp(0.0, 1.0)
        pred_depth_map = depth_map.clone()
        rgb_map, depth_map = (
            rgb_map.reshape(H, W, 3).cpu(),
            depth_map.reshape(H, W).cpu(),
        )

        if cfg.data.tool_mask:
            tool_mask = data["tool_mask"]

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)
        if "depths" in data.keys():
            depth = data["depths"]
            gt_depth, _ = visualize_depth_numpy(depth.numpy(), near_far)

        if len(test_dataset):
            gt_rgb = gt_rgb.view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                ms_ssim = MS_SSIM(
                    rgb_map.permute(2, 0, 1).unsqueeze(0),
                    gt_rgb.permute(2, 0, 1).unsqueeze(0),
                    data_range=1,
                    size_average=True,
                )
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), "alex", device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), "vgg", device)

                pred_depth_map = pred_depth_map.reshape(H, W).cpu()
                gt_depth_map = depth.reshape(H, W).cpu()
                pred_depth_map = np.nan_to_num(pred_depth_map).astype(np.float32)
                gt_depth_map = np.nan_to_num(gt_depth_map).astype(np.float32)
                depth_error = torch.mean(
                    torch.abs(torch.tensor(pred_depth_map) - torch.tensor(gt_depth_map))
                )
                depth_errors.append(depth_error)

                ssims.append(ssim)
                msssims.append(ms_ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype("uint8")
        gt_rgb_map = (gt_rgb.numpy() * 255).astype("uint8")

        if depth is not None:
            gt_depth_maps.append(gt_depth)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        gt_rgb_maps.append(gt_rgb_map)
        if savePath is not None:
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}.png", rgb_map)
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_gt.png", gt_rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f"{savePath}/rgbd/{prefix}{idx:03d}.png", rgb_map)
            if depth is not None:
                rgb_map = np.concatenate((gt_rgb_map, gt_depth), axis=1)
                imageio.imwrite(f"{savePath}/rgbd/{prefix}{idx:03d}_gt.png", rgb_map)

        if cfg.optim.optimize_poses:
            c2w = torch.cat(
                [poses_rot.detach().cpu(), poses_t[..., None].detach().cpu()], dim=-1
            )
            all_poses_pred = torch.cat([c2w, c2w[idx * 8][None]], dim=0)
            colours = ["C1"] * c2w.shape[0] + ["C2"]
            # visualize only optimized poses
            all_poses = all_poses_pred
            # make it more visually distinguishable
            if cfg.local.use_preprocessed_poses:
                all_poses[:, :3, 3] *= 50.0
            pose_vis = draw_poses(all_poses.cpu(), colours)
            poses_vis.append(pose_vis)

    imageio.mimwrite(
        f"{savePath}/{prefix}video.mp4",
        np.stack(rgb_maps),
        fps=8,
        format="FFMPEG",
        quality=8,
    )
    imageio.mimwrite(
        f"{savePath}/{prefix}gt_video.mp4",
        np.stack(gt_rgb_maps),
        fps=8,
        format="FFMPEG",
        quality=8,
    )
    imageio.mimwrite(
        f"{savePath}/{prefix}depthvideo.mp4",
        np.stack(depth_maps),
        format="FFMPEG",
        fps=8,
        quality=8,
    )
    if depth is not None:
        imageio.mimwrite(
            f"{savePath}/{prefix}gt_depthvideo.mp4",
            np.stack(gt_depth_maps),
            format="FFMPEG",
            fps=30,
            quality=10,
        )
    if cfg.optim.optimize_poses:
        with open(f"{savePath}/{prefix}posesvideo.mp4", "wb") as f:
            imageio.mimwrite(
                f,
                np.stack(poses_vis),
                fps=8,
                quality=6,
                format="mp4",
                output_params=["-f", "mp4"],
            )  # fps initially 30

    imgs_show = []
    for i_frame in trange(len(rgb_maps), desc="DEMO|Save 2D results", leave=False):
        out_show = (
            np.hstack(
                [
                    add_text_to_img(gt_rgb_maps[i_frame], "rgb_gt"),
                    add_text_to_img(rgb_maps[i_frame], "rgb_pred"),
                ]
            ),
        )
        imgs_show.append(out_show)
    # Generate gif
    tqdm.write("DEMO|Generate rendering gif...")
    fps = 8
    with iio.get_writer(
        f"{savePath}/{prefix}.gif", mode="I", duration=1 / fps
    ) as writer:
        for frame in imgs_show:
            writer.append_data(frame[0])

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            msssim = np.mean(np.asarray(msssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            depth_error = np.mean(np.asarray(depth_errors))
            with open(f"{savePath}/{prefix}mean.txt", "w") as f:
                f.write(
                    f"PSNR: {psnr}, SSIM: {ssim}, MS-SSIM: {msssim}, LPIPS_a: {l_a}, LPIPS_v: {l_v}, L1-Dist: {depth_error}, L1-Dist(mm): {depth_error * test_dataset.original_depth_scale}\n"
                )
                print(
                    f"PSNR: {psnr}, SSIM: {ssim}, MS-SSIM: {msssim}, LPIPS_a: {l_a}, LPIPS_v: {l_v}, L1-Dist: {depth_error}, L1-Dist(mm): {depth_error * test_dataset.original_depth_scale}\n"
                )
                for i in range(len(PSNRs)):
                    f.write(
                        f"Index {i}, PSNR: {PSNRs[i]}, SSIM: {ssims[i]}, MS-SSIM: {msssim}, LPIPS_a: {l_alex[i]}, LPIPS_v: {l_vgg[i]}, L1-Dist: {depth_errors[i]}, L1-Dist(mm): {depth_errors[i] * test_dataset.original_depth_scale}\n"
                    )
        else:
            with open(f"{savePath}/{prefix}mean.txt", "w") as f:
                f.write(f"PSNR: {psnr} \n")
                print(f"PSNR: {psnr} \n")
                for i in range(len(PSNRs)):
                    f.write(f"Index {i}, PSNR: {PSNRs[i]}\n")

    return PSNRs


@torch.no_grad()
def evaluation_path(
    test_dataset,
    model,
    cfg,
    savePath=None,
    N_vis=5,
    prefix="",
    N_samples=-1,
    white_bg=False,
    ndc_ray=False,
    compute_extra_metrics=True,
    device="cuda",
):
    """
    Evaluate the model on the valiation rays.
    """
    rgb_maps, depth_maps = [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    val_rays, val_times = test_dataset.get_val_rays()

    for idx in tqdm(range(val_times.shape[0])):
        W, H = test_dataset.img_wh
        rays = val_rays[idx]
        time = val_times[idx]
        time = time.expand(rays.shape[0], 1)
        rgb_map, _, depth_map, _, _, _ = OctreeRender_trilinear_fast(
            rays,
            time,
            model,
            chunk=8192,
            N_samples=N_samples,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
        )
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = (
            rgb_map.reshape(H, W, 3).cpu(),
            depth_map.reshape(H, W).cpu(),
        )

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)

        rgb_map = (rgb_map.numpy() * 255).astype("uint8")

        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}.png", rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f"{savePath}/rgbd/{prefix}{idx:03d}.png", rgb_map)

    imageio.mimwrite(
        f"{savePath}/{prefix}video.mp4", np.stack(rgb_maps), fps=4, quality=8
    )
    imageio.mimwrite(
        f"{savePath}/{prefix}depthvideo.mp4", np.stack(depth_maps), fps=4, quality=8
    )

    return 0


@torch.no_grad()
def evaluation_local(
    test_dataset,
    train_dataset,
    modelsdirectory,
    cfg,
    savePath=None,
    N_vis=5,
    prefix="",
    N_samples=-1,
    white_bg=False,
    ndc_ray=False,
    compute_extra_metrics=True,
    device="cuda",
    image_bounds=None,
    world2hexs=None,
    num_train_images=None,
    poses_rot=None,
    poses_t=None,
    blending_weights=None,
):
    """
    Evaluate the model on the test rays and compute metrics.
    """

    if cfg.only_pose_metrics:
        os.makedirs(savePath + "/poses", exist_ok=True)
        if not cfg.optim.optimize_poses:
            if cfg.local.use_preprocessed_poses:
                c2w = torch.cat(
                    [poses_rot.detach().cpu(), poses_t[..., None].detach().cpu()],
                    dim=-1,
                )
            else:
                c2w = get_cam2world(poses_rot.detach().cpu(), poses_t.detach().cpu())
        else:
            c2w = get_cam2world(poses_rot.detach().cpu(), poses_t.detach().cpu())

        test_idxs = np.arange(0, len(c2w))
        if not os.path.exists(os.path.join(cfg.data.datadir, "groundtruth.txt")):
            print("Groundtruth pose file does not exist!")
            sys.exit()
        (
            ate_rmse,
            rpe_trans_mean,
            rpe_rot_mean,
            trans_error,
            rpe_trans,
            rpe_rot,
        ) = eval_poses(
            data_dir=cfg.data.datadir,
            depth_scale=test_dataset.depth_scale_factor,
            test_idxs=test_idxs,
            pred_list=c2w,
            delta=1,
            offset=0,
            savePath=savePath,
        )

        # save metrics!
        with open(f"{savePath}/{prefix}_pose_mean.txt", "w") as f:
            f.write(
                f"ATE-RMSE: {ate_rmse}, Trans-Error: {trans_error.mean()}, RPE-Trans: {rpe_trans_mean}, RPE-Rot: {rpe_rot_mean}\n"
            )
            print(
                f"ATE-RMSE: {ate_rmse}, Trans-Error: {trans_error.mean()}, RPE-Trans: {rpe_trans_mean}, RPE-Rot: {rpe_rot_mean}"
            )
            for i in range(len(trans_error)):
                if i < (len(trans_error) - 1):
                    f.write(
                        f"Index {i}, Trans-Error: {trans_error[i]}, RPE-Trans: {rpe_trans[i]}, RPE-Rot: {rpe_rot[i]}\n"
                    )
                else:
                    f.write(f"Index {i}, Trans-Error: {trans_error[i]}\n")

        (
            ate_rmse,
            rpe_trans_mean,
            rpe_rot_mean,
            trans_error,
            rpe_trans,
            rpe_rot,
        ) = eval_poses(
            data_dir=cfg.data.datadir,
            depth_scale=1.0,
            test_idxs=test_idxs,
            pred_list=None,
            delta=1,
            offset=0,
        )

        # save metrics!
        if cfg.local.use_preprocessed_poses:
            with open(f"{savePath}/{prefix}_preprocessed_pose_mean.txt", "w") as f:
                f.write(
                    f"ATE-RMSE: {ate_rmse}, Trans-Error: {trans_error.mean()}, RPE-Trans: {rpe_trans_mean}, RPE-Rot: {rpe_rot_mean}\n"
                )
                print(
                    f"ATE-RMSE: {ate_rmse}, Trans-Error: {trans_error.mean()}, RPE-Trans: {rpe_trans_mean}, RPE-Rot: {rpe_rot_mean}"
                )
                for i in range(len(trans_error)):
                    if i < (len(trans_error) - 1):
                        f.write(
                            f"Index {i}, Trans-Error: {trans_error[i]}, RPE-Trans: {rpe_trans[i]}, RPE-Rot: {rpe_rot[i]}\n"
                        )
                    else:
                        f.write(f"Index {i}, Trans-Error: {trans_error[i]}\n")
        else:
            print("Baseline Results:\n")
            print(
                f"ATE-RMSE: {ate_rmse}, Trans-Error: {trans_error.mean()}, RPE-Trans: {rpe_trans_mean}, RPE-Rot: {rpe_rot_mean}"
            )

        sys.exit()

    near_far = test_dataset.near_far

    if cfg.test_weights:
        test_weights_local(
            test_dataset,
            N_samples,
            ndc_ray,
            white_bg,
            device,
            near_far,
            savePath,
            cfg=cfg,
            blending_weights=blending_weights,
            poses_rot=poses_rot,
            poses_t=poses_t,
            world2hexs=world2hexs,
            modelsdirectory=modelsdirectory,
        )
        sys.exit()

    # generate entirely new views and possibly also pc's
    if cfg.render_extra:
        render_extra_local(
            test_dataset,
            N_samples,
            ndc_ray,
            white_bg,
            device,
            near_far,
            cfg.pc_test_idx,
            savePath,
            prefix,
            repeats=cfg.render_extra_repeats,
            num_scene=cfg.num_scene,
            original_pose=cfg.original_pose,
            gen_pc=cfg.create_pc,
            cfg=cfg,
            blending_weights=blending_weights,
            poses_rot=poses_rot,
            poses_t=poses_t,
            world2hexs=world2hexs,
            modelsdirectory=modelsdirectory,
        )
        sys.exit()

    # generate pc's only for predefined set of test views
    if cfg.create_pc and not cfg.render_extra:
        create_pc_local(
            test_dataset,
            N_samples,
            ndc_ray,
            white_bg,
            device,
            cfg.pc_test_idx,
            savePath,
            prefix,
            cfg,
            blending_weights=blending_weights,
            poses_rot=poses_rot,
            poses_t=poses_t,
            world2hexs=world2hexs,
            modelsdirectory=modelsdirectory,
        )
        sys.exit()

    if cfg.render_new_poses:
        render_new_poses(
            test_dataset,
            N_samples,
            ndc_ray,
            white_bg,
            device,
            near_far=near_far,
            new_pose_file=cfg.new_pose_file,
            poses_rot=poses_rot,
            poses_t=poses_t,
            savePath=savePath,
            prefix=prefix,
            cfg=cfg,
            blending_weights=blending_weights,
            world2hexs=world2hexs,
            modelsdirectory=modelsdirectory,
        )
        sys.exit()

    PSNRs, rgb_maps, depth_maps, gt_depth_maps, gt_rgb_maps, poses_vis = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    msssims, ssims, l_alex, l_vgg, depth_errors = [], [], [], [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/rgbd", exist_ok=True)
    os.makedirs(savePath + "/poses", exist_ok=True)
    try:
        tqdm._instances.clear()
    except Exception:
        pass

    print(f"Image Bound: {image_bounds}")

    # find max global image
    total_frames = len(test_dataset) + len(train_dataset.poses)
    all_pre_poses, train_img_idxs = get_all_poses(
        train_dataset, test_dataset, total_frames
    )
    global_bounds = map_train_bounds2global(image_bounds[-1], train_img_idxs)
    upper_img_bound = global_bounds[-1]
    print(f"Upper Image Bound: {upper_img_bound}")

    idxs = np.arange(0, len(test_dataset))
    for idx in tqdm(idxs):
        data = test_dataset[idx]
        if cfg.local_models and cfg.local.progressive_opt:
            # stop early when not enough models are trained!!!
            if (idx * 8) > upper_img_bound:
                break
            W, H = test_dataset.img_wh
            ray_idx = torch.arange(0, int(W * H))
            i, j = ids2pixel(W, H, ray_idx)
            directions = get_ray_directions_lean(
                i,
                j,
                [test_dataset.focal, test_dataset.focal],
                [test_dataset.cx, test_dataset.cy],
            )
            # directions = get_ray_directions_lean(i, j, [test_dataset.focal, test_dataset.focal], [W/2, H/2])
            # directions = directions / torch.norm(directions, dim=-1, keepdim=True)
            idx_repeat = (torch.ones(directions.shape[0]) * idx * 8).to(torch.long)
            if cfg.local.use_preprocessed_poses:
                cam2world = torch.cat(
                    [
                        poses_rot[idx_repeat].detach().cpu(),
                        poses_t[idx_repeat][..., None].detach().cpu(),
                    ],
                    dim=-1,
                )
            else:
                cam2world = get_cam2world(
                    poses_rot.detach().cpu(), poses_t.detach().cpu(), idx_repeat
                )
            cam2hexs = {}
            active_hex_ids = torch.nonzero(blending_weights[idx * 8, :])[:, 0].tolist()

            for model_id in active_hex_ids:
                cam2hex = cam2world.clone()
                cam2hex[:, :3, 3] += world2hexs[model_id].detach().cpu()

                cam2hexs[model_id] = cam2hex
        else:
            samples = data["rays"]

        gt_rgb, sample_times = data["rgbs"], data["time"]
        gt_depth_map = data["depths"]

        W, H = test_dataset.img_wh

        # init variables
        rgb_map = torch.zeros_like(directions, device=device)
        depth_map = torch.zeros_like(directions[..., 0], device=device)

        # reshape blending weights
        blending_weights_expanded = blending_weights[idx_repeat]

        for model_id in active_hex_ids:
            blending_weight = blending_weights_expanded[:, model_id]
            cam2hex = cam2hexs[model_id]
            model = torch.load(f"{modelsdirectory}_{model_id}.th", map_location=device)
            samples = get_rays_lean(directions, cam2hex)
            rays = samples.view(-1, samples.shape[-1])
            times = sample_times.view(-1, sample_times.shape[-1])
            (
                rgb_map_t,
                _,
                depth_map_t,
                _,
                xyz_sampled_t,
                weights_t,
            ) = OctreeRender_trilinear_fast(
                rays,
                times,
                model,
                chunk=cfg.optim.batch_size // 2,
                N_samples=N_samples,
                ndc_ray=ndc_ray,
                white_bg=white_bg,
                device=device,
            )
            # adjust according to blending weights
            rgb_map = rgb_map + rgb_map_t * blending_weight[..., None]
            depth_map = depth_map + depth_map_t * blending_weight

        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_map, depth_map = (
            rgb_map.reshape(H, W, 3).cpu(),
            depth_map.reshape(H, W).cpu(),
        )
        torch.cuda.empty_cache()

        if cfg.data.tool_mask:
            tool_mask = data["tool_mask"]

        depth_map_vis, _ = visualize_depth_numpy(depth_map.numpy(), near_far)
        if "depths" in data.keys():
            depth = data["depths"]
            gt_depth_vis, _ = visualize_depth_numpy(depth.numpy(), near_far)

        if len(test_dataset):
            gt_rgb = gt_rgb.view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                ms_ssim = MS_SSIM(
                    rgb_map.permute(2, 0, 1).unsqueeze(0),
                    gt_rgb.permute(2, 0, 1).unsqueeze(0),
                    data_range=1,
                    size_average=True,
                )
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), "alex", device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), "vgg", device)

                depth_map = depth_map.reshape(H, W).cpu()
                gt_depth_map = gt_depth_map.reshape(H, W).cpu()
                depth_map = np.nan_to_num(depth_map).astype(np.float32)
                gt_depth = np.nan_to_num(gt_depth_map).astype(np.float32)
                depth_error = torch.mean(
                    torch.abs(torch.tensor(depth_map) - torch.tensor(gt_depth_map))
                )
                depth_errors.append(depth_error)

                ssims.append(ssim)
                msssims.append(ms_ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype("uint8")
        gt_rgb_map = (gt_rgb.numpy() * 255).astype("uint8")

        if depth is not None:
            gt_depth_maps.append(gt_depth_vis)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map_vis)
        gt_rgb_maps.append(gt_rgb_map)
        if savePath is not None:
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}.png", rgb_map)
            imageio.imwrite(f"{savePath}/{prefix}{idx:03d}_gt.png", gt_rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map_vis), axis=1)
            imageio.imwrite(f"{savePath}/rgbd/{prefix}{idx:03d}.png", rgb_map)
            if depth is not None:
                rgb_map = np.concatenate((gt_rgb_map, gt_depth_vis), axis=1)
                imageio.imwrite(f"{savePath}/rgbd/{prefix}{idx:03d}_gt.png", rgb_map)

        if cfg.optim.optimize_poses:
            if cfg.local.use_preprocessed_poses:
                gt_c2w = all_pre_poses
                all_pre_poses_gt = torch.cat([gt_c2w, gt_c2w[idx * 8][None]], dim=0)[
                    :, :3, :4
                ]
                c2w = torch.cat(
                    [poses_rot.detach().cpu(), poses_t[..., None].detach().cpu()],
                    dim=-1,
                )
            else:
                c2w = get_cam2world(poses_rot.detach().cpu(), poses_t.detach().cpu())
            all_poses_pred = torch.cat([c2w, c2w[idx * 8][None]], dim=0)
            colours = ["C1"] * c2w.shape[0] + ["C2"]
            if cfg.local.use_preprocessed_poses:
                # visualize optimized poses and preprocessed poses
                all_poses = torch.cat([all_poses_pred, all_pre_poses_gt], dim=0)
                colours = colours + gt_c2w.shape[0] * ["C3"] + ["C4"]
            else:
                # visualize only optimized poses
                all_poses = all_poses_pred

            # make it more visually distinguishable
            if cfg.local.use_preprocessed_poses:
                all_poses[:, :3, 3] *= 50.0
            pose_vis = draw_poses(all_poses.cpu(), colours)
            # pose_vis = cv2.resize(pose_vis, (int(pose_vis.shape[1] * rgb_map.shape[0] / pose_vis.shape[0]), rgb_map.shape[0]))
            poses_vis.append(pose_vis)

    imageio.mimwrite(
        f"{savePath}/{prefix}video.mp4",
        np.stack(rgb_maps),
        fps=8,
        format="FFMPEG",
        quality=8,
    )
    imageio.mimwrite(
        f"{savePath}/{prefix}_gt_video.mp4",
        np.stack(gt_rgb_maps),
        fps=8,
        format="FFMPEG",
        quality=8,
    )
    imageio.mimwrite(
        f"{savePath}/{prefix}depthvideo.mp4",
        np.stack(depth_maps),
        format="FFMPEG",
        fps=8,
        quality=8,
    )
    if depth is not None:
        imageio.mimwrite(
            f"{savePath}/{prefix}_gt_depthvideo.mp4",
            np.stack(gt_depth_maps),
            format="FFMPEG",
            fps=30,
            quality=10,
        )

    if cfg.local.progressive_opt and os.path.exists(
        os.path.join(cfg.data.datadir, "groundtruth.txt")
    ):
        if not cfg.optim.optimize_poses:
            c2w = torch.cat(
                [poses_rot.detach().cpu(), poses_t[..., None].detach().cpu()], dim=-1
            )

        test_idxs = np.arange(0, len(c2w))
        (
            ate_rmse,
            rpe_trans_mean,
            rpe_rot_mean,
            trans_error,
            rpe_trans,
            rpe_rot,
        ) = eval_poses(
            data_dir=cfg.data.datadir,
            depth_scale=test_dataset.depth_scale_factor,
            test_idxs=test_idxs,
            pred_list=c2w,
            delta=1,
            offset=0,
            savePath=savePath,
        )

        # save metrics!
        with open(f"{savePath}/{prefix}_pose_mean.txt", "w") as f:
            f.write(
                f"ATE-RMSE: {ate_rmse}, Trans-Error: {trans_error.mean()}, RPE-Trans: {rpe_trans_mean}, RPE-Rot: {rpe_rot_mean}\n"
            )
            print(
                f"ATE-RMSE: {ate_rmse}, Trans-Error: {trans_error.mean()}, RPE-Trans: {rpe_trans_mean}, RPE-Rot: {rpe_rot_mean}"
            )
            for i in range(len(trans_error)):
                if i < (len(trans_error) - 1):
                    f.write(
                        f"Index {i}, Trans-Error: {trans_error[i]}, RPE-Trans: {rpe_trans[i]}, RPE-Rot: {rpe_rot[i]}\n"
                    )
                else:
                    f.write(f"Index {i}, Trans-Error: {trans_error[i]}\n")

        (
            ate_rmse,
            rpe_trans_mean,
            rpe_rot_mean,
            trans_error,
            rpe_trans,
            rpe_rot,
        ) = eval_poses(
            data_dir=cfg.data.datadir,
            depth_scale=1.0,
            test_idxs=test_idxs,
            pred_list=None,
            delta=1,
            offset=0,
        )

        # save metrics!
        # if cfg.local.use_preprocessed_poses:
        with open(f"{savePath}/{prefix}_preprocessed_pose_mean.txt", "w") as f:
            f.write(
                f"ATE-RMSE: {ate_rmse}, Trans-Error: {trans_error.mean()}, RPE-Trans: {rpe_trans_mean}, RPE-Rot: {rpe_rot_mean}\n"
            )
            print(
                f"ATE-RMSE: {ate_rmse}, Trans-Error: {trans_error.mean()}, RPE-Trans: {rpe_trans_mean}, RPE-Rot: {rpe_rot_mean}"
            )
            for i in range(len(trans_error)):
                if i < (len(trans_error) - 1):
                    f.write(
                        f"Index {i}, Trans-Error: {trans_error[i]}, RPE-Trans: {rpe_trans[i]}, RPE-Rot: {rpe_rot[i]}\n"
                    )
                else:
                    f.write(f"Index {i}, Trans-Error: {trans_error[i]}\n")

    if len(poses_vis) > 0:
        with open(f"{savePath}/{prefix}posesvideo.mp4", "wb") as f:
            imageio.mimwrite(
                f,
                np.stack(poses_vis),
                fps=8,
                quality=6,
                format="mp4",
                output_params=["-f", "mp4"],
            )  # fps initially 30

    imgs_show = []
    for i_frame in trange(len(rgb_maps), desc="DEMO|Save 2D results", leave=False):
        out_show = (
            np.hstack(
                [
                    add_text_to_img(gt_rgb_maps[i_frame], "rgb_gt"),
                    add_text_to_img(rgb_maps[i_frame], "rgb_pred"),
                ]
            ),
        )
        imgs_show.append(out_show)
    # Generate gif
    tqdm.write("DEMO|Generate rendering gif...")
    fps = 8
    with iio.get_writer(
        f"{savePath}/{prefix}.gif", mode="I", duration=1 / fps
    ) as writer:
        for frame in imgs_show:
            writer.append_data(frame[0])

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            msssim = np.mean(np.asarray(msssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            depth_error = np.mean(np.asarray(depth_errors))
            with open(f"{savePath}/{prefix}mean.txt", "w") as f:
                f.write(
                    f"PSNR: {psnr}, SSIM: {ssim}, MS-SSIM: {msssim}, LPIPS_a: {l_a}, LPIPS_v: {l_v}, L1-Dist: {depth_error}, L1-Dist(mm): {depth_error * test_dataset.original_depth_scale}\n"
                )
                print(
                    f"PSNR: {psnr}, SSIM: {ssim}, MS-SSIM: {msssim}, LPIPS_a: {l_a}, LPIPS_v: {l_v}, L1-Dist: {depth_error}, L1-Dist(mm): {depth_error * test_dataset.original_depth_scale}\n"
                )
                for i in range(len(PSNRs)):
                    f.write(
                        f"Index {i}, PSNR: {PSNRs[i]}, SSIM: {ssims[i]}, MS-SSIM: {msssim}, LPIPS_a: {l_alex[i]}, LPIPS_v: {l_vgg[i]}, L1-Dist: {depth_errors[i]}, L1-Dist(mm): {depth_errors[i] * test_dataset.original_depth_scale}\n"
                    )
        else:
            with open(f"{savePath}/{prefix}mean.txt", "w") as f:
                f.write(f"PSNR: {psnr} \n")
                print(f"PSNR: {psnr} \n")
                for i in range(len(PSNRs)):
                    f.write(f"Index {i}, PSNR: {PSNRs[i]}\n")

    return PSNRs
