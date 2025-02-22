import sys
import torch
import numpy as np
import math
from typing import Union, Tuple
import os
from lietorch import SE3
from modulefinder import Module
from flex.render.util.util import draw_poses

import imageio.v2 as iio
import cv2
import imageio


def get_vector_angle(vector1, vector2):

    vector1 = vector1[3:]-vector1[:3]
    vector2 = vector2[3:]-vector2[:3]
    
    angle = ((torch.dot(vector1,vector2))/(torch.linalg.vector_norm(vector1.float(),ord=2).item()*torch.linalg.vector_norm(vector2.float(),ord=2).item())).item()
    angle = torch.arccos(torch.tensor([angle])).item() # angles in radians
    angle = angle * 180/math.pi # angles in degree
    

    return angle


def get_angle(pose1, pose2):

    R1 = pose1[:3,:3]
    R2 = pose2[:3,:3]
    rotation_diff = np.dot(R1, R2.T)
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    # Avoid invalid values due to numerical errors
    error_cos = min(1.0, max(-1.0, 0.5 * (trace - 1.0)))
    rd_deg = np.rad2deg(np.arccos(error_cos))
    #es.append(rd_deg)
    angle = rd_deg

    return angle


def compute_bbox(all_rays, near, far):
    print("compute_bbox_by_cam_frustrm: start")
    world_bound_scale = 1.1
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    rays_o = all_rays[:,0:3]
    viewdirs = all_rays[:,3:6]

    pts_nf = torch.stack(
        [rays_o + viewdirs * near, rays_o + viewdirs * far]
    )
    
    xyz_min = torch.minimum(xyz_min, pts_nf.amin((0, 1)))
    xyz_max = torch.maximum(xyz_max, pts_nf.amax((0, 1)))
    print("compute_bbox_by_cam_frustrm: xyz_min", xyz_min)
    print("compute_bbox_by_cam_frustrm: xyz_max", xyz_max)
    print("compute_bbox_by_cam_frustrm: finish")
    xyz_shift = (xyz_max - xyz_min) * (world_bound_scale - 1) / 2
    xyz_min -= xyz_shift
    xyz_max += xyz_shift
    return xyz_min, xyz_max


def preprocess_partitioning(args, train_dataset, device):
    image_bounds = []
    world2hex = []
    rel_pose = []
    #r_c2w = []
    #t_c2w = []
    all_rays = train_dataset.all_rays.clone().detach()
    all_poses = train_dataset.poses.clone().detach()
    W, H = train_dataset.img_wh
    focal = train_dataset.focal
    center = [train_dataset.cx, train_dataset.cy]
    num_images = int(all_rays.shape[0]/(W*H))
    print(num_images)
    #directions = get_ray_directions(H, W, focal, center)
    print("Poses Shape: "+str(all_poses.shape))

    for i in range(num_images):
        inv_pose = torch.linalg.inv(all_poses[i].clone().detach())

        if len(image_bounds)==0:
            image_bounds.append([0,i])
            #world2hex.append(torch.tensor([0,0,0]))
            world2hex.append(all_poses[0,:3,3].clone().detach())
            world2hex_pose = all_poses[0].clone().detach()
        
        elif i-image_bounds[-1][0]>=args.local.n_max_frames or torch.norm((inv_pose @ world2hex_pose)[:3,3]) > args.local.max_drift or abs(get_angle(all_poses[i].clone().detach(), world2hex_pose.clone())) >= args.local.angle_threshold:
            print("Relative distance: "+str(torch.norm((inv_pose @ world2hex_pose)[:3,3])))
            print("Angle: "+str(get_angle(all_poses[i].clone().detach(), world2hex_pose.clone())))
            image_bounds[-1][1] = i
            if i<(num_images-1):
                n_overlap = min(args.local.n_overlap, (image_bounds[-1][1]-image_bounds[-1][0]))
                image_bounds.append([i-n_overlap,i])
                world2hex.append(all_poses[i,:3,3].clone().detach())
                world2hex_pose = all_poses[i].clone().detach()
            else:
                image_bounds[-1][1] = num_images
        elif i>=(num_images-1):
            image_bounds[-1][1] = num_images


    aabb_lists = []

    # calc. local bboxes
    print(image_bounds)
    idx = 0
    for item in image_bounds:
        rays = all_rays[(item[0]*H*W):(item[1]*H*W)]
        aabb_lists.append(torch.stack((compute_bbox(rays, near=args.data.near, far=args.data.far)),dim=0))
        idx += 1
    print(aabb_lists)
    print("Num. models to train: "+str(len(image_bounds)))

    return aabb_lists, image_bounds, world2hex


def generate_rel_poses(poses, device):

    rel_poses = []
    poses = poses.detach().cpu().numpy()
    if poses is not None:
        for idx in range(len(poses)):
            if idx == 0:
                pose = np.eye(4, dtype=np.float32)
            else:
                pose = np.linalg.inv(poses[idx - 1]) @ poses[idx]
            rel_poses.append(pose)
        rel_poses = np.stack(rel_poses, axis=0) 

        scale = 1.0

    return torch.tensor(rel_poses).to(device), scale


def sixD_to_mtx(r):
    b1 = r[..., 0]
    b1 = b1 / torch.norm(b1, dim=-1)[:, None]
    b2 = r[..., 1] - torch.sum(b1 * r[..., 1], dim=-1)[:, None] * b1
    b2 = b2 / torch.norm(b2, dim=-1)[:, None]
    b3 = torch.cross(b1, b2)

    return torch.stack([b1, b2, b3], dim=-1)


def mtx_to_sixD(r):
    return torch.stack([r[..., 0], r[..., 1]], dim=-1)


def eval_poses(data_dir:str, pred_list:torch.Tensor, depth_scale, test_idxs, delta:int=1, offset:int=0, ret_align_T=False, ignore_failed_pos=False, savePath="",gen_images=False):

    # load gt poses:
    gt_list = read_poses(pose_path=os.path.join(data_dir, 'groundtruth.txt'))

    if pred_list is None:
        #calc. input poses metrics
        pred_list = read_poses(pose_path=os.path.join(data_dir, 'trajectory.freiburg'))
        preprocessed = True
    else:
        # Bring pred poses back into OpenCV format and undo depth normalization
        flip = np.eye(4)
        flip[1,1] = -1
        flip[2,2] = -1
        pred_list = np.matmul(pred_list, flip)
        pred_list = pred_list.detach().cpu().numpy()
        preprocessed = False


    pred_poses = []
    gt_poses = []

    for k in test_idxs:
        if (k+offset >= 0) & (k+offset < len(gt_list)):
            pred_poses.append(pred_list[k])
            gt_poses.append(gt_list[k+offset])
    

    pred_poses = np.stack(pred_poses)
    gt_poses = np.stack(gt_poses)
    if not preprocessed:
        pred_poses = np.concatenate([pred_poses, np.array([0,0,0,1]).reshape(1, 1, 4).repeat(pred_poses.shape[0], axis=0)], axis=1)
    if gen_images:
        preprocessed=False
    ate_rmse, trans_error, transform , valid= absolute_trajectory_error(gt_poses.copy(), pred_poses.copy(), ret_align_T=True,
                                                                 ignore_failed_pos=ignore_failed_pos, preprocessed=preprocessed, savePath=savePath)

    rpe_trans, rpe_rot = relative_pose_error(gt_poses, pred_poses, delta=delta, ignore_failed_pos=ignore_failed_pos)
    if ret_align_T:
        return ate_rmse, np.mean(rpe_trans), np.mean(rpe_rot), trans_error, rpe_trans, rpe_rot, transform, gt_poses, valid
    return ate_rmse, np.mean(rpe_trans), np.mean(rpe_rot), trans_error, rpe_trans, rpe_rot


def align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Estimate similarity transformation (rotation, scale, translation) from source to target (such as the Sim3 group).
    """
    
    k, n = source.shape

    mx = source.mean(axis=1)
    my = target.mean(axis=1)
    source_centered = source - np.tile(mx, (n, 1)).T
    target_centered = target - np.tile(my, (n, 1)).T

    sx = np.mean(np.sum(source_centered**2, axis=0))
    sy = np.mean(np.sum(target_centered**2, axis=0))

    Sxy = (target_centered @ source_centered.T) / n

    U, D, Vt = np.linalg.svd(Sxy, full_matrices=True, compute_uv=True)
    V = Vt.T
    rank = np.linalg.matrix_rank(Sxy)
    if rank < k:
        raise ValueError("Failed to estimate similarity transformation")

    S = np.eye(k)
    if np.linalg.det(Sxy) < 0:
        S[k - 1, k - 1] = -1

    R = U @ S @ V.T

    s = np.trace(np.diag(D) @ S) / sx
    t = my - s * (R @ mx)

    return R, s, t


def absolute_trajectory_error(gt_poses: Union[np.ndarray, torch.Tensor],
                              predicted_poses: Union[np.ndarray, torch.Tensor],
                              prealign: bool=True, ret_align_T: bool=False,
                              ignore_failed_pos: bool=False,
                              preprocessed: bool=False,
                              savePath: str = "") -> Tuple[float, Union[np.ndarray, torch.Tensor]]:
    """
        Absolute Trajectory Error ATE-RMSE

    :param gt_poses: array of ground-truth poses, shape nx4x4
    :param predicted_poses: array of estimated poses, shape nx4x4
    :param prealign: if true, pre-align trajectories using method of Horn
    :return: ate-RMSE, translation errors
    :rtype: float
    """
    assert len(gt_poses) == len(predicted_poses)
    lib = get_lib(gt_poses)
    # ignore identity pose predictions as these mark failed pose estimations
    valid = [True]
    if ignore_failed_pos:
        for i in range(len(predicted_poses) - 1):
            valid.append((predicted_poses[i] - predicted_poses[i + 1]).sum() != 0)
    else:
        valid = lib.ones(len(predicted_poses), dtype=bool)
    T = None
    if prealign:

        R, s, t = align(predicted_poses[valid, :3, 3].T, gt_poses[valid, :3, 3].T)

        predicted_poses[:,:3,3] = predicted_poses[:,:3,3]*s
        #predicted_poses[:,:3,3] = (R @ predicted_poses[:,:3,3].T + t.reshape(-1, 1)).T
        T = np.eye(4)
        T[:3,:3] = R
        T[:3, 3] = t
        predicted_poses = T[None, ...] @ predicted_poses

        # visualize alignment
        all_poses_pred = torch.cat([torch.tensor(predicted_poses), torch.tensor(predicted_poses[0][None])], dim=0)
        colours_pred = ["C1"] * predicted_poses.shape[0] + ["C2"]
        # visualize optimized poses and preprocessed poses
        all_poses = torch.cat([all_poses_pred, torch.tensor(gt_poses)], dim=0)
        colours = colours_pred + gt_poses.shape[0] * ["C3"]
        pose_vis = draw_poses(all_poses.cpu().float(), colours)
        pred_pose_vis = draw_poses(all_poses_pred[:-1].cpu().float(), colours_pred[:-1])
        colours_gt = gt_poses.shape[0] * ["C1"]
        gt_pose_vis = draw_poses(torch.tensor(gt_poses).float(), colours_gt)

        if not preprocessed:
            imageio.imwrite(f"{savePath}/poses_both.png", np.stack(pose_vis))
            imageio.imwrite(f"{savePath}/pred_poses.png", np.stack(pred_pose_vis))
            imageio.imwrite(f"{savePath}/gt_poses.png", np.stack(gt_pose_vis))


    trans_err = []
    for gt, pred, v in zip(gt_poses, predicted_poses, valid):
        if v:
            trans_err.append(lib.sum((gt[:3,3].T-pred[:3, 3])**2))
    trans_err = np.asarray(trans_err)
    ate_pos = lib.sqrt(lib.mean(trans_err))
    if ret_align_T:
        return ate_pos, np.sqrt(trans_err), T, valid
    return ate_pos, np.sqrt(trans_err)


def relative_pose_error(gt_poses: Union[np.ndarray, torch.Tensor],
                        predicted_poses: Union[np.ndarray, torch.Tensor],
                        delta: int=1,
                        ignore_failed_pos: bool=False):
    """
            Relative Pose Error RPE (mean)

        :param gt_poses: array of ground-truth poses, shape nx4x4
        :param predicted_poses: array of estimated poses, shape nx4x4
        :param delta: time interval to compute relative poses
        :return: rpe-trans, rpe-rot
        :rtype: float
        """
    assert len(gt_poses) == len(predicted_poses)
    lib = get_lib(gt_poses)

    # ignore identity pose predictions as these mark failed pose estimations
    valid = [True]
    if ignore_failed_pos:
        for i in range(len(predicted_poses) - 1):
            valid.append((predicted_poses[i] - predicted_poses[i + 1]).sum() != 0)
    else:
        valid = lib.ones(len(predicted_poses), dtype=bool)
    
    # pre-align
    R, s, t = align(predicted_poses[valid, :3, 3].T, gt_poses[valid, :3, 3].T)

    predicted_poses[:,:3,3] = predicted_poses[:,:3,3]*s
    #predicted_poses[:,:3,3] = (R @ predicted_poses[:,:3,3].T + t.reshape(-1, 1)).T
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = t
    predicted_poses = T[None, ...] @ predicted_poses

    trans_errors = []
    rot_errors = []
    for i in range(len(gt_poses)-delta):
        if ((predicted_poses[i] - predicted_poses[i+1]).sum() != 0) | (not ignore_failed_pos):
            gt_rel = np.linalg.inv(gt_poses[i]) @ gt_poses[i+delta]
            pred_rel = np.linalg.inv(predicted_poses[i]) @ predicted_poses[i+delta]
            rel_err = np.linalg.inv(gt_rel) @ pred_rel

            trans_errors.append(lib.sqrt(lib.sum((rel_err[:3,3])**2)))
            d = 0.5*(lib.trace(rel_err[:3, :3]) - 1)
            rot_errors.append(np.arccos(max(min(d, 1.0), -1.0)))

    rpe_trans = np.asarray(trans_errors)
    rpe_rot = np.asarray(rot_errors)
    return rpe_trans, rpe_rot

def end_trajectory_error(predicted_poses: Union[np.ndarray, torch.Tensor], savePath: str):
    # compute relative errors between first and last pose
    # only meaningful when no gt poses available and first frame is identical to last frame in pred seq 
    # visualize alignment
    all_poses_pred = torch.cat([torch.tensor(predicted_poses), torch.tensor(predicted_poses[0][None])], dim=0)
    all_poses_pred = torch.cat([all_poses_pred, torch.tensor(predicted_poses[-1][None])], dim=0)
    colours_pred = ["tab:orange"] * predicted_poses.shape[0] + ["tab:green"] + ["tab:red"]
    # visualize optimized poses and preprocessed poses
    pred_pose_vis = draw_poses(all_poses_pred.cpu().float(), colours_pred)
    imageio.imwrite(f"{savePath}/pred_poses.png", np.stack(pred_pose_vis))

    predicted_poses = np.concatenate([predicted_poses, np.array([0,0,0,1]).reshape(1, 1, 4).repeat(predicted_poses.shape[0], axis=0)], axis=1)
    length = len(predicted_poses)
    first_half_poses = predicted_poses[:(length//2)]
    second_half_poses = predicted_poses[(length//2):]
    rev_count = -1
    lib = get_lib(predicted_poses)
    rpe_trans_list, rpe_rot_list = [], []
    for i in range(len(second_half_poses)):
        start_pose = first_half_poses[rev_count]
        end_pose = second_half_poses[i]
        rel_err = np.linalg.inv(start_pose) @ end_pose
        rpe_trans = lib.sqrt(lib.sum((rel_err[:3,3])**2))
        d = 0.5*(lib.trace(rel_err[:3,:3]) - 1)
        rpe_rot = np.arccos(max(min(d, 1.0), -1.0))
        rpe_trans_list.append(rpe_trans)
        rpe_rot_list.append(rpe_rot)
        rev_count-=1

    return rpe_trans_list, rpe_rot_list


def icosphere(nu = 1, nr_verts = None):
    '''
    Returns a geodesic icosahedron with subdivision frequency nu. Frequency
    nu = 1 returns regular unit icosahedron, and nu>1 preformes subdivision.
    If nr_verts is given, nu will be adjusted such that icosphere contains
    at least nr_verts vertices. Returned faces are zero-indexed!
        
    Parameters
    ----------
    nu : subdivision frequency, integer (larger than 1 to make a change).
    nr_verts: desired number of mesh vertices, if given, nu may be increased.
        
    
    Returns
    -------
    subvertices : vertex list, numpy array of shape (20+10*(nu+1)*(nu-1)/2, 3)
    subfaces : face list, numpy array of shape (10*n**2, 3)
    
    '''
  
    # Unit icosahedron
    (vertices,faces) = icosahedron()

    # If nr_verts given, computing appropriate subdivision frequency nu.
    # We know nr_verts = 12+10*(nu+1)(nu-1)
    if not nr_verts is None:
        nu_min = np.ceil(np.sqrt(max(1+(nr_verts-12)/10, 1)))
        nu = max(nu, nu_min)
        
    # Subdividing  
    if nu>1:
        (vertices,faces) = subdivide_mesh(vertices, faces, nu)
        vertices = vertices/np.sqrt(np.sum(vertices**2, axis=1, keepdims=True))

    return(vertices, faces)

def icosahedron():
    '''' Regular unit icosahedron. '''
    
    # 12 principal directions in 3D space: points on an unit icosahedron
    phi = (1+np.sqrt(5))/2    
    vertices = np.array([
        [0, 1, phi], [0,-1, phi], [1, phi, 0],
        [-1, phi, 0], [phi, 0, 1], [-phi, 0, 1]])/np.sqrt(1+phi**2)
    vertices = np.r_[vertices,-vertices]
    
    # 20 faces
    faces = np.array([
        [0,5,1], [0,3,5], [0,2,3], [0,4,2], [0,1,4], 
        [1,5,8], [5,3,10], [3,2,7], [2,4,11], [4,1,9], 
        [7,11,6], [11,9,6], [9,8,6], [8,10,6], [10,7,6], 
        [2,11,7], [4,9,11], [1,8,9], [5,10,8], [3,7,10]], dtype=int)    
    
    return (vertices, faces)


def subdivide_mesh(vertices, faces, nu):
    '''
    Subdivides mesh by adding vertices on mesh edges and faces. Each edge 
    will be divided in nu segments. (For example, for nu=2 one vertex is added  
    on each mesh edge, for nu=3 two vertices are added on each mesh edge and 
    one vertex is added on each face.) If V and F are number of mesh vertices
    and number of mesh faces for the input mesh, the subdivided mesh contains 
    V + F*(nu+1)*(nu-1)/2 vertices and F*nu^2 faces.
    
    Parameters
    ----------
    vertices : vertex list, numpy array of shape (V,3) 
    faces : face list, numby array of shape (F,3). Zero indexed.
    nu : subdivision frequency, integer (larger than 1 to make a change).
    
    Returns
    -------
    subvertices : vertex list, numpy array of shape (V + F*(nu+1)*(nu-1)/2, 3)
    subfaces : face list, numpy array of shape (F*n**2, 3)
    
    Author: vand at dtu.dk, 8.12.2017. Translated to python 6.4.2021
    
    '''
        
    edges = np.r_[faces[:,:-1], faces[:,1:],faces[:,[0,2]]]
    edges = np.unique(np.sort(edges, axis=1),axis=0)
    F = faces.shape[0]
    V = vertices.shape[0]
    E = edges.shape[0] 
    subfaces = np.empty((F*nu**2, 3), dtype = int)
    subvertices = np.empty((V+E*(nu-1)+F*(nu-1)*(nu-2)//2, 3))
                        
    subvertices[:V] = vertices
    
    # Dictionary for accessing edge index from indices of edge vertices.
    edge_indices = dict()
    for i in range(V):
        edge_indices[i] = dict()
    for i in range(E):
        edge_indices[edges[i,0]][edges[i,1]] = i
        edge_indices[edges[i,1]][edges[i,0]] = -i
         
    template = faces_template(nu)
    ordering = vertex_ordering(nu)
    reordered_template = ordering[template]
    
    # At this point, we have V vertices, and now we add (nu-1) vertex per edge
    # (on-edge vertices).
    w = np.arange(1,nu)/nu # interpolation weights
    for e in range(E):
        edge = edges[e]
        for k in range(nu-1):
            subvertices[V+e*(nu-1)+k] = (w[-1-k] * vertices[edge[0]] 
                                         + w[k] * vertices[edge[1]])
  
    # At this point we have E(nu-1)+V vertices, and we add (nu-1)*(nu-2)/2 
    # vertices per face (on-face vertices).
    r = np.arange(nu-1)
    for f in range(F):
        # First, fixing connectivity. We get hold of the indices of all
        # vertices invoved in this subface: original, on-edges and on-faces.
        T = np.arange(f*(nu-1)*(nu-2)//2+E*(nu-1)+V, 
                      (f+1)*(nu-1)*(nu-2)//2+E*(nu-1)+V) # will be added
        eAB = edge_indices[faces[f,0]][faces[f,1]] 
        eAC = edge_indices[faces[f,0]][faces[f,2]] 
        eBC = edge_indices[faces[f,1]][faces[f,2]] 
        AB = reverse(abs(eAB)*(nu-1)+V+r, eAB<0) # already added
        AC = reverse(abs(eAC)*(nu-1)+V+r, eAC<0) # already added
        BC = reverse(abs(eBC)*(nu-1)+V+r, eBC<0) # already added
        VEF = np.r_[faces[f], AB, AC, BC, T]
        subfaces[f*nu**2:(f+1)*nu**2, :] = VEF[reordered_template]
        # Now geometry, computing positions of face vertices.
        subvertices[T,:] = inside_points(subvertices[AB,:],subvertices[AC,:])
    
    return (subvertices, subfaces)

def reverse(vector, flag): 
    '''' For reversing the direction of an edge. ''' 
    
    if flag:
        vector = vector[::-1]
    return(vector)


def faces_template(nu):
    '''
    Template for linking subfaces                  0
    in a subdivision of a face.                   / \
    Returns faces with vertex                    1---2
    indexing given by reading order             / \ / \
    (as illustratated).                        3---4---5
                                              / \ / \ / \
                                             6---7---8---9    
                                            / \ / \ / \ / \ 
                                           10--11--12--13--14 
    '''
  
    faces = []
    # looping in layers of triangles
    for i in range(nu): 
        vertex0 = i*(i+1)//2
        skip = i+1      
        for j in range(i): # adding pairs of triangles, will not run for i==0
            faces.append([j+vertex0, j+vertex0+skip, j+vertex0+skip+1])
            faces.append([j+vertex0, j+vertex0+skip+1, j+vertex0+1])
        # adding the last (unpaired, rightmost) triangle
        faces.append([i+vertex0, i+vertex0+skip, i+vertex0+skip+1])
        
    return (np.array(faces))


def vertex_ordering(nu):
    ''' 
    Permutation for ordering of                    0
    face vertices which transformes               / \
    reading-order indexing into indexing         3---6
    first corners vertices, then on-edges       / \ / \
    vertices, and then on-face vertices        4---12--7
    (as illustrated).                         / \ / \ / \
                                             5---13--14--8
                                            / \ / \ / \ / \ 
                                           1---9--10--11---2 
    '''
    
    left = [j for j in range(3, nu+2)]
    right = [j for j in range(nu+2, 2*nu+1)]
    bottom = [j for j in range(2*nu+1, 3*nu)]
    inside = [j for j in range(3*nu,(nu+1)*(nu+2)//2)]
    
    o = [0] # topmost corner
    for i in range(nu-1):
        o.append(left[i])
        o = o + inside[i*(i-1)//2:i*(i+1)//2]
        o.append(right[i])
    o = o + [1] + bottom + [2]
        
    return(np.array(o))


def total_trajectory_length(gt_list: list):

    locs = np.stack([g.translation().numpy() for g in gt_list])
    translations = np.sqrt(np.sum(np.diff(locs, axis=0)**2, axis=-1))
    return np.sum(translations)


def read_freiburg(path: str, ret_stamps=False, no_stamp=False):
    with open(path, 'r') as f:
        data = f.read()
        lines = data.replace(",", " ").replace("\t", " ").split("\n")
        list = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
                len(line) > 0 and line[0] != "#"]
    if no_stamp:
        trans = torch.from_numpy(np.asarray([l[0:3] for l in list if len(l) > 0], dtype=float))
        trans *= 1000.0  #m to mm
        quat = torch.from_numpy(np.asarray([l[3:] for l in list if len(l) > 0], dtype=float))
        pose_se3 = SE3.InitFromVec(torch.cat((trans, quat), dim=-1))
    else:
        time_stamp = [l[0] for l in list if len(l) > 0]
        try:
            time_stamp = np.asarray([int(l.split('.')[0] + l.split('.')[1]) for l in time_stamp])*100
        except IndexError:
            time_stamp = np.asarray([int(l) for l in time_stamp])
        trans = torch.from_numpy(np.asarray([l[1:4] for l in list if len(l) > 0], dtype=float))
        trans *= 1000.0  # m to mm
        quat = torch.from_numpy(np.asarray([l[4:] for l in list if len(l) > 0], dtype=float))
        pose_se3 = SE3.InitFromVec(torch.cat((trans, quat), dim=-1))
        if ret_stamps:
            return pose_se3, time_stamp

    return pose_se3


def read_poses(pose_path, return_stamps=False):

    if not isinstance(pose_path, dict):
            pose_list, pose_stamps = read_freiburg(pose_path, ret_stamps=True)
            pose_list = {key: pose for key, pose in zip(pose_stamps, pose_list)}
    else:
        pose_list = pose_path

    pose_keys = sorted(list(pose_list.keys()))
    poses = []
    for k in pose_keys:
        poses.append(pose_list[k].matrix())
    poses = np.stack(poses)
    if return_stamps:
        return poses, pose_stamps
    else:
        return poses


def get_lib(
        data_object: Union[np.ndarray, torch.Tensor]
    ) -> Module:

    if isinstance(data_object, np.ndarray):
        return np
    
    if isinstance(data_object, torch.Tensor):
        return torch

    raise TypeError('%s is not supported' % type(data_object))