from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class System_Config:
    seed: int = 20220401
    basedir: str = "./log"
    ckpt: Optional[str] = None
    progress_refresh_rate: int = 100
    vis_every: int = 10000
    save_ckpt_every: int = 10000
    add_timestamp: bool = True
    localrf2_ckpt: str = 'logs/localrf_test_2/miti_23.th'


@dataclass
class Model_Config:
    model_name: str = "HexPlane_Slim"  # choose from "HexPlane", "HexPlane_Slim"
    N_voxel_init: int = 64 * 64 * 64  # initial voxel number
    N_voxel_final: int = 200 * 200 * 200  # final voxel number
    step_ratio: float = 0.5
    nonsquare_voxel: bool = True  # if yes, voxel numbers along each axis depend on scene length along each axis
    time_grid_init: int = 16
    time_grid_final: int = 128
    normalize_type: str = "normal"
    upsample_list: List[int] = field(default_factory=lambda: [3000, 6000, 9000])
    update_emptymask_list: List[int] = field(
        default_factory=lambda: [4000, 8000, 10000]
    )

    # Plane Initialization
    density_n_comp: List[int] = field(default_factory=lambda: [24, 24, 24])
    app_n_comp: List[int] = field(default_factory=lambda: [48, 48, 48])
    density_dim: int = 1
    app_dim: int = 27
    DensityMode: str = "plain"  # choose from "plain", "general_MLP"
    AppMode: str = "general_MLP"
    init_scale: float = 0.1
    init_shift: float = 0.0

    # Fusion Methods
    fusion_one: str = "multiply"
    fusion_two: str = "concat"

    # Density Feature Settings
    fea2denseAct: str = "softplus"
    density_shift: float = -10.0
    distance_scale: float = 25.0

    # Density Regressor MLP settings
    density_t_pe: int = -1
    density_pos_pe: int = -1
    density_view_pe: int = -1
    density_fea_pe: int = 2
    density_featureC: int = 128
    density_n_layers: int = 3

    # Appearance Regressor MLP settings
    app_t_pe: int = -1
    app_pos_pe: int = -1
    app_view_pe: int = 2
    app_fea_pe: int = 2
    app_featureC: int = 128
    app_n_layers: int = 3

    # Empty mask settings
    emptyMask_thres: float = 0.001
    rayMarch_weight_thres: float = 0.0001
    no_emptyMask: bool = False

    # Reg
    rgb_loss_weight: float = 1.0 #3.0 DynamicNeRF #1.0 HexPlane
    random_background: bool = False
    depth_loss: bool = True
    urf_depth_loss_weight: float = 0.0 # urban radiance field depth loss
    optical_flow_loss_weight: float = 1.0 # loss weight of optical flow supervision

    TV_t_s_ratio: float = 2.0  # ratio of TV loss along temporal and spatial dimensions
    TV_weight_density: float = 0.0001
    TV_weight_app: float = 0.0001
    L1_weight_density: float = 0.0
    L1_weight_app: float = 0.0

    # Sampling
    align_corners: bool = True
    # There are two types of upsampling: aligned and unaligned.
    # Aligned upsampling: N_t = 2 * N_t-1 - 1. Like: |--|--| ->upsampling -> |-|-|-|-|, where | represents sampling points and - represents distance.
    # Unaligned upsampling: We use linear_interpolation to get the new sampling points without considering the above rule.
    # using "unaligned" upsampling will essentially double the grid sizes at each time, ignoring N_voxel_final.
    upsampling_type: str = "aligned"  # choose from "aligned", "unaligned".
    nSamples: int = 1000000

    # Flow
    predict_flow: bool = False
    opt_from_depth: bool = True # induce optical flow from predicted depth only --> no scene flow required!

    # pose optim
    barf_c2f: List[float] = field(default_factory=lambda: [0.0, 0.0]) # in SPARF [0.4,0.7] # for activating PE when optimising for poses

    # Contract scene
    contract: bool = False # contract scene


@dataclass
class Data_Config:
    datadir: str = "./data"
    dataset_name: str = "dnerf"  # choose from "dnerf", "neural3D_NDC"
    downsample: float = 1.0
    cal_fine_bbox: bool = False
    N_vis: int = -1
    time_scale: float = 1.0
    scene_bbox_min: List[float] = field(default_factory=lambda: [-1.0, -1.0, -1.0])
    scene_bbox_max: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    N_random_pose: int = 1000
    near: float = 0.01 # near plane for ray sampling
    far: float = 1.0 # far plane for ray sampling
    depth_data: bool = True # using depth supervision
    flow_data : bool = False # provide optical flow data from RAFT
    tool_mask: bool = False # provide surgical tool mask
    use_ndc: bool = False # map rays into NDC space

    # for dnerf

    # for neural3D_NDC
    nv3d_ndc_bd_factor: float = 0.75
    nv3d_ndc_eval_step: int = 1
    nv3d_ndc_eval_index: int = 0
    nv3d_ndc_sphere_scale: float = 1.0

    # Hierachical Sampling for Neural3D_NDC
    multi_cam: bool = False
    stage_1_iteration: int = 300000
    stage_2_iteration: int = 250000
    stage_3_iteration: int = 100000
    key_f_num: int = 10
    stage_1_gamma: float = 0.001
    stage_2_gamma: float = 0.02
    stage_3_alpha: float = 0.1

    datasampler_type: str = "rays"  # choose from "rays", "images", "hierach"


@dataclass
class Optim_Config:
    # Learning Rate
    lr_density_grid: float = 0.02
    lr_app_grid: float = 0.02
    lr_density_nn: float = 0.001
    lr_app_nn: float = 0.001
    lr_feat_nn: float = 0.001

    # Optimizer, Adam deault
    beta1: float = 0.9
    beta2: float = 0.99
    lr_decay_type: str = "exp"  # choose from "exp" or "cosine" or "linear" or "cyclic"
    lr_decay_target_ratio: float = 0.1
    lr_decay_step: int = -1
    lr_upsample_reset: bool = True

    batch_size: int = 4096
    n_iters: int = 25000
    num_s_imgs: int = 1 # number of images in each batch

    # Pose optimization
    optimize_poses: bool = False
    lr_R_init: float = 5e-3 # initial lr for pose rotation
    lr_T_init: float = 5e-4 # initial lr for pose translation

    # Optical flow
    temp_decline: int = 10 # equivalent to reducing flow loss weight by 10 every 10% of training setting it to 5 will reduced loss weight every 20% --> set to 1 when not reducing at all
    
    # Cyclic LR scheduler
    cyc_step_size: int = 25000
    cyc_gamma: float = 0.99
    cyc_mode: str = "triangular" # options: "triangular" or "triangular2" or "exp_range



@dataclass
class Local_Config:
    # LocalHex settings
    n_max_frames: int = 100 # max amount of frames per model
    n_overlap: int = 30 # max frames overlap between models
    max_drift: float = 1.0 # min distance between poses for generating new model
    angle_threshold: float = 10.0 # min angle between poses for generating new model
    progressive_opt: bool = False # no prepartitioning of models but train on the fly
    n_init_frames: int = 5 # initial amount of frames for progressive optimization per model
    add_frames_every: int = 100 # iterations before adding new frame
    use_preprocessed_poses: bool = True # use preprocessed poses else start from identity pose
    use_camera_momentum: bool = False # use last camera change to initialize new pose


@dataclass
class Config:
    config: Optional[str] = None
    expname: str = "default"

    render_only: bool = False
    render_train: bool = False
    render_test: bool = True
    render_path: bool = False
    render_new_poses: bool = False
    new_pose_file: str = "test_poses.freiburg"

    render_extra: bool = False # render novel views different to train/test views
    original_pose: bool = False # keep pose from specific scene for all timesteps
    num_scene: int = 0 # test view from which to pick the original pose
    render_extra_repeats: int = 14 # number of different poses for each timestep
    img_limit: int = 200 # number of original images from test dataset to render for in render_extra if bigger than len(dataset) take full test dataset
    eval_every_img: int = 1 # indicates which img idxs are evaluated for e.g. when this is 1 then every test img is selected if it is 5 every fifth test image is selected

    create_pc: bool = False # create pc's from model rgb and depth predictions
    pc_test_idx: List[int] = field(default_factory=lambda: [0]) # scene numbers for which to create pc's
    pc_icp: bool = False # run icp algorithm on predicted pc and gt pc
    compute_pc_cd: bool = False # compute L1-depth loss on estimated depth for test views 

    test_weights: bool = False # plot density weights distribution across several timesteps 
    test_view: int = 10
    ray_idxs: List[int] = field(default_factory=lambda: [10000, 20000, 50000, 90000, 120000])

    visualize_4d_vol: bool = False # visualize ground truth 4d volume avg and variance per ray only
    vis_occupancy: bool = False # visualize ocpacity std along each ray from model opacity predictions based on 4d volume 

    only_pose_metrics: bool = False

    # local models
    local_models: bool = False # split scene into smaller segments and train one model each

    # Debug Mode
    debug_mode: bool = False

    systems: System_Config = field(default_factory=lambda :System_Config())
    model: Model_Config = field(default_factory=lambda : Model_Config())
    data: Data_Config = field(default_factory=lambda : Data_Config())
    optim: Optim_Config = field(default_factory=lambda : Optim_Config())
    local: Local_Config = field(default_factory=lambda : Local_Config())

