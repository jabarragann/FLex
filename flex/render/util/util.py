import math

import cv2
import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
import copy


def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """

    x = np.nan_to_num(depth)  # change nan to 0
    if minmax is None:
        mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi, ma = minmax

    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi, ma]


def N_to_reso(n_voxels, bbox, adjusted_grid=True):
    if adjusted_grid:
        xyz_min, xyz_max = bbox
        voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / 3)
        return ((xyz_max - xyz_min) / voxel_size).long().tolist()
    else:
        # grid_each = n_voxels.pow(1 / 3)
        grid_each = math.pow(n_voxels, 1 / 3)
        return [int(grid_each), int(grid_each), int(grid_each)]


def decode_flow(encoded_flow):
    flow = encoded_flow[..., :2].astype(np.float32)
    flow -= 2**15
    flow /= 2**8
    return flow, (encoded_flow[..., 2] > 2**15).astype(np.float32)

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def pts2px(pts, f, center):
    pts[..., 1] = -pts[..., 1]
    pts[..., 2] = -pts[..., 2]
    pts[..., 2] = torch.clip(pts[..., 2].clone(), min=1e-6)
    return torch.stack(
        [pts[..., 0] / pts[..., 2] * f + center[0] - 0.5, pts[..., 1] / pts[..., 2] * f + center[1] - 0.5], 
        dim=-1)

def inverse_pose(pose):
    pose_inv = torch.zeros_like(pose)
    pose_inv[:, :3, :3] = torch.transpose(pose[:, :3, :3], 1, 2)
    pose_inv[:, :3, 3] = -torch.bmm(pose_inv[:, :3, :3].clone(), pose[:, :3, 3:])[..., 0]
    return pose_inv

def get_cam2cams(cam2worlds, indices, offset):
    idx = torch.clamp(indices + offset, 0, len(cam2worlds) - 1)
    world2cam = inverse_pose(cam2worlds[idx])
    cam2cams = torch.zeros_like(world2cam)
    cam2cams[:, :3, :3] = torch.bmm(world2cam[:, :3, :3], cam2worlds[indices, :3, :3])
    cam2cams[:, :3, 3] = torch.bmm(world2cam[:, :3, :3], cam2worlds[indices, :3, 3:])[..., 0]
    cam2cams[:, :3, 3] += world2cam[:, :3, 3]
    return cam2cams

def get_fwd_bwd_cam2cams(cam2worlds, indices):
    fwd_cam2cams = get_cam2cams(cam2worlds, indices, 1)
    bwd_cam2cams = get_cam2cams(cam2worlds, indices, -1)
    return fwd_cam2cams, bwd_cam2cams

def get_pred_flow(pts, ij, cam2cams, focal, center):
    new_pts = torch.transpose(torch.bmm(cam2cams[:, :3, :3], torch.transpose(pts, 1, 2)), 1, 2)
    new_pts = new_pts + cam2cams[:, None, :3, 3]
    new_ij = pts2px(new_pts, focal, center)

    return new_ij - ij.float(), new_ij

def get_all_poses(train_dataset, test_dataset, total_frames):

    all_poses = torch.zeros((total_frames, 4, 4))
    # find original test and train view idxs
    num_train_images = total_frames-len(test_dataset)
    test_num = np.arange(len(test_dataset))
    orig_test_num = test_num*8
    train_images = np.arange(num_train_images)
    for i in range(len(train_images)):
        for test_img in orig_test_num:
            if train_images[i] >= test_img:
                train_images[i] += 1
            else:
                break

    all_poses[orig_test_num] = test_dataset.poses
    all_poses[train_images] = train_dataset.poses

    return all_poses, train_images

def get_adjacent_test_train_idxs(train_img_idxs, total_frames):
    
    prev_frame_idxs, next_frame_idxs = [], []
    num_test_frames = total_frames-len(train_img_idxs)
    test_frames = torch.arange(0, num_test_frames)*8

    for i in test_frames:
        next_frame_idx = (i+1==torch.tensor(train_img_idxs)).nonzero()[0,0].item()
        if i>0:
            prev_frame_idx = (i-1==torch.tensor(train_img_idxs)).nonzero()[0,0].item()
        else:
            prev_frame_idx = 0

        prev_frame_idxs.append(prev_frame_idx)
        next_frame_idxs.append(next_frame_idx)

    return prev_frame_idxs, next_frame_idxs


def L1(x, M=None):
    if M == None:
        return torch.mean(torch.abs(x))
    else:
        return torch.sum(torch.abs(x) * M) / (torch.sum(M) + 1e-8) / x.shape[-1]

def L2(x, M=None):
    if M == None:
        return torch.mean(x ** 2)
    else:
        return torch.sum((x ** 2) * M) / (torch.sum(M) + 1e-8) / x.shape[-1]


def resize_flow(flow, W_new, H_new):
    H_old, W_old = flow.shape[0:2]
    flow_resized = cv2.resize(flow, (W_new, H_new), interpolation=cv2.INTER_LINEAR)
    flow_resized = flow_resized.astype(float)
    flow_resized[:, :, 0] *= H_new / H_old
    flow_resized[:, :, 1] *= W_new / W_old
    return flow_resized

def map_train_bounds2global(train_bounds, train_img_idxs):

    global_bounds = [train_img_idxs[train_bounds[0]], train_img_idxs[train_bounds[1]-1]+1]
    if global_bounds[0]==1: #include first test frame also
        global_bounds[0]=0

    return global_bounds


def draw_poses(poses, colours):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    centered_poses = poses.clone()
    centered_poses[:, :, 3] -= torch.mean(centered_poses[:, :, 3], dim=0, keepdim=True)

    vertices, faces, wireframe = get_camera_mesh(
        centered_poses, 0.05
    )
    center = vertices[:, -1]
    ps = max(torch.max(center).item(), 0.1)
    ms = min(torch.min(center).item(), -0.1)
    ax.set_xlim3d(ms, ps)
    ax.set_ylim3d(ms, ps)
    ax.set_zlim3d(ms, ps)
    wireframe_merged = merge_wireframes(wireframe)
    for c in range(center.shape[0]):
        ax.plot(
            wireframe_merged[0][c * 10 : (c + 1) * 10],
            wireframe_merged[1][c * 10 : (c + 1) * 10],
            wireframe_merged[2][c * 10 : (c + 1) * 10],
            color=colours[c],
        )

    plt.tight_layout()
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img # np.zeros([5, 5, 3], dtype=np.uint8)

def get_camera_mesh(pose, depth=1):
    vertices = (
        torch.tensor(
            [[-0.5, -0.5, -1], [0.5, -0.5, -1], [0.5, 0.5, -1], [-0.5, 0.5, -1], [0, 0, 0]]
        )
        * depth
    )
    faces = torch.tensor(
        [[0, 1, 2], [0, 2, 3], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
    )
    vertices = vertices @ pose[:, :3, :3].transpose(-1, -2)
    vertices += pose[:, None, :3, 3]
    vertices[..., 1:] *= -1 # Axis flip
    wireframe = vertices[:, [0, 1, 2, 3, 0, 4, 1, 2, 4, 3]]
    return vertices, faces, wireframe


def merge_wireframes(wireframe):
    wireframe_merged = [[], [], []]
    for w in wireframe:
        wireframe_merged[0] += [float(n) for n in w[:, 0]]
        wireframe_merged[1] += [float(n) for n in w[:, 1]]
        wireframe_merged[2] += [float(n) for n in w[:, 2]]
    return wireframe_merged


def compute_2d_weights(weights_p, 
                    occ_weights_p):
    weight_map_p = torch.sum(weights_p.detach() * (1.0 - occ_weights_p), -1)
    return weight_map_p

def to8b(x):
    if torch.is_tensor(x):
        x = tensor2array(x)
    return (255.*np.clip(x,0,1)).astype(np.uint8)


def add_text_to_img(img,
                    text,
                    font=cv2.FONT_HERSHEY_SIMPLEX,
                    font_color=(255,0,0),
                    org=(10,50),
                    font_scale=2,
                    thickness=4,
                    line_type=cv2.LINE_AA,
                    ):
    if torch.is_tensor(img):
        img = tensor2array(img)
    if not np.issubdtype(img.dtype, np.uint8):
        img = to8b(img)
    img_show = cv2.putText(copy.deepcopy(img), text, org, font, font_scale, 
                           font_color, thickness, line_type)
    return img_show