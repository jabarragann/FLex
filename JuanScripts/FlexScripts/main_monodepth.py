import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import datetime
import os
import random
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from config.config import Config
from flex.dataloader import get_test_dataset, get_train_dataset
from flex.model import init_model
from flex.render.mono_depth_trainer import MonoDepthTrainer
from flex.render.render import evaluation, evaluation_local, evaluation_path
from flex.render.util.util import get_all_poses
from JuanScripts.FlexScripts.test_functions import juan_pose_evaluations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)


def render_test(cfg):
    test_dataset = get_test_dataset(cfg, is_stack=True)
    if (
        not cfg.only_pose_metrics
        and not cfg.test_weights
        and not cfg.render_extra
        and not cfg.render_new_poses
    ):
        train_dataset = get_train_dataset(cfg, is_stack=True)
    else:
        train_dataset = None
    ndc_ray = test_dataset.ndc_ray
    white_bg = test_dataset.white_bg
    assert cfg.model.predict_flow == cfg.data.flow_data

    if cfg.local_models:
        logfolder = os.path.dirname(cfg.systems.ckpt)
        # modelsdirectory = (cfg.systems.ckpt).split(".")[0] # A bit hacky --> TODO: make this easier
        modelsdirectory = cfg.systems.ckpt + "/" + cfg.expname

        image_bounds = torch.load(modelsdirectory + "_image_bounds.th")
        world2hexs = torch.load(modelsdirectory + "_world2hexs.th")
        if cfg.local.progressive_opt:
            poses_rot = torch.load(modelsdirectory + "_poses_rot.th")
            poses_t = torch.load(modelsdirectory + "_poses_t.th")
            # poses_rot = poses_rot[:101]
            # poses_t = poses_t[:101]
            blending_weights = torch.load(modelsdirectory + "_blending_weights.th")
        else:
            poses_rot, poses_t = None, None

        if True:  # Juan Tests
            print("Juan Tests")
            juan_pose_evaluations(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                poses_t=poses_t,
                poses_rot=poses_rot,
                save_path=cfg.systems.ckpt + "/juan_evaluations/",
            )

        if cfg.render_test:  # Original FLex test
            os.makedirs(f"{logfolder}/imgs_test_all", exist_ok=True)
            evaluation_local(
                test_dataset,
                train_dataset,
                modelsdirectory,
                cfg,
                cfg.systems.ckpt + "/imgs_test_all/",
                prefix="test",
                N_vis=-1,
                N_samples=-1,
                ndc_ray=ndc_ray,
                white_bg=white_bg,
                device=device,
                image_bounds=image_bounds,
                world2hexs=world2hexs,
                num_train_images=image_bounds[-1][1],
                poses_rot=poses_rot,
                poses_t=poses_t,
                blending_weights=blending_weights,
            )

    # else:
    #     if not os.path.exists(cfg.systems.ckpt):
    #         print("the ckpt path does not exists!!")
    #         print(f"{cfg.systems.ckpt}")
    #         return

    #     HexPlane = torch.load(cfg.systems.ckpt, map_location=device)
    #     logfolder = os.path.dirname(cfg.systems.ckpt)

    #     all_poses, train_images_idxs = get_all_poses(
    #         train_dataset, test_dataset, len(test_dataset) + len(train_dataset)
    #     )

    #     if cfg.render_train:
    #         os.makedirs(f"{logfolder}/imgs_train_all", exist_ok=True)
    #         train_dataset = get_train_dataset(cfg, is_stack=True)
    #         evaluation(
    #             train_dataset,
    #             HexPlane,
    #             cfg,
    #             f"{logfolder}/imgs_train_all/",
    #             prefix="train",
    #             N_vis=-1,
    #             N_samples=-1,
    #             ndc_ray=ndc_ray,
    #             white_bg=white_bg,
    #             poses_rot=all_poses[:, :3, :3],
    #             poses_t=all_poses[:, :3, 3],
    #             device=device,
    #         )

    #     if cfg.render_test:
    #         os.makedirs(f"{logfolder}/imgs_test_all", exist_ok=True)
    #         evaluation(
    #             test_dataset,
    #             HexPlane,
    #             cfg,
    #             f"{logfolder}/imgs_test_all/",
    #             prefix="test",
    #             N_vis=-1,
    #             N_samples=-1,
    #             ndc_ray=ndc_ray,
    #             white_bg=white_bg,
    #             poses_rot=all_poses[:, :3, :3],
    #             poses_t=all_poses[:, :3, 3],
    #             device=device,
    #         )

    #     if cfg.render_path:
    #         os.makedirs(f"{logfolder}/imgs_path_all", exist_ok=True)
    #         evaluation_path(
    #             test_dataset,
    #             HexPlane,
    #             cfg,
    #             f"{logfolder}/imgs_path_all/",
    #             prefix="test",
    #             N_vis=-1,
    #             N_samples=-1,
    #             ndc_ray=ndc_ray,
    #             white_bg=white_bg,
    #             device=device,
    #         )


def progressive_opt(train_dataset, test_dataset, cfg, summary_writer, logfolder):
    aabb = train_dataset.scene_bbox.to(device)
    first_HexPlane, reso_cur = init_model(cfg, aabb, test_dataset.near_far, device)
    start_time = time.time()
    # init trainer
    trainer = MonoDepthTrainer(
        first_HexPlane,
        cfg,
        reso_cur,
        train_dataset,
        test_dataset,
        summary_writer,
        logfolder,
        device,
    )

    trainer.train()
    print(f"Training time: {(time.time() - start_time) / 60} min.")

    # Save rotation and translation matrices:
    torch.save(
        torch.stack(list(trainer.poses_rot), 0),
        f"{logfolder}/{cfg.expname}_poses_rot.th",
    )
    torch.save(
        torch.stack(list(trainer.poses_t), 0),
        f"{logfolder}/{cfg.expname}_poses_t.th",
    )
    # save image bounds and world2hex coordinates:
    torch.save(
        torch.tensor(trainer.image_bounds),
        f"{logfolder}/{cfg.expname}_image_bounds.th",
    )
    torch.save(
        torch.stack(list(trainer.world2hexs), 0),
        f"{logfolder}/{cfg.expname}_world2hexs.th",
    )
    # save blending weights:
    torch.save(
        torch.stack(list(trainer.blending_weights), 0),
        f"{logfolder}/{cfg.expname}_blending_weights.th",
    )
    # for configuration save initial model again --> not really needed
    torch.save(first_HexPlane, f"{logfolder}/{cfg.expname}.th")

    return trainer


def reconstruction_local(cfg):
    if cfg.local.progressive_opt:
        # must be pre-defined with fixed bound due to pose optimization
        # ORIGINAL
        # if cfg.data.cal_fine_bbox != False:
        #     cfg.data.cal_fine_bbox = False
        # modified
        cfg.data.cal_fine_bbox = False

    if cfg.data.datasampler_type == "rays":
        train_dataset = get_train_dataset(cfg, is_stack=False)
    else:
        train_dataset = get_train_dataset(cfg, is_stack=True)

    test_dataset = get_test_dataset(cfg, is_stack=True)
    ndc_ray = test_dataset.ndc_ray
    white_bg = test_dataset.white_bg
    # near_far = test_dataset.near_far

    assert cfg.model.predict_flow == cfg.data.flow_data

    if cfg.systems.add_timestamp:
        logfolder = f"{cfg.systems.basedir}/{cfg.expname}{datetime.datetime.now().strftime('-%Y%m%d-%H%M%S')}"
    else:
        logfolder = f"{cfg.systems.basedir}/{cfg.expname}"

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f"{logfolder}/imgs_vis", exist_ok=True)
    os.makedirs(f"{logfolder}/imgs_rgba", exist_ok=True)
    os.makedirs(f"{logfolder}/rgba", exist_ok=True)

    summary_writer = SummaryWriter(os.path.join(logfolder, "logs"))
    cfg_file = os.path.join(f"{logfolder}", "cfg.yaml")
    with open(cfg_file, "w") as f:
        OmegaConf.save(config=cfg, f=f)

    if cfg.local.progressive_opt:
        trainer = progressive_opt(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            cfg=cfg,
            summary_writer=summary_writer,
            logfolder=logfolder,
        )
    else:
        raise NotImplementedError("Only local progressive optimization is implemented.")

    total_params = 0
    for model_id in range(0, len(trainer.hexplanes)):
        total_params += sum(
            p.numel()
            for p in trainer.hexplanes[model_id].parameters()
            if p.requires_grad
        )

    print(f"Model contains {total_params / 1000000} Mil. parameters")

    # Render test viewpoints.
    modelsdirectory = f"{logfolder}/{cfg.expname}"
    if cfg.render_test:
        os.makedirs(f"{logfolder}/imgs_test_all", exist_ok=True)
        evaluation_local(
            test_dataset,
            train_dataset,
            modelsdirectory,
            cfg,
            f"{logfolder}/{cfg.expname}/imgs_test_all/",
            prefix="test",
            N_vis=-1,
            N_samples=-1,
            ndc_ray=ndc_ray,
            white_bg=white_bg,
            device=device,
            image_bounds=trainer.image_bounds,
            world2hexs=trainer.world2hexs,
            num_train_images=trainer.image_bounds[-1][1],
            poses_rot=(
                torch.stack(list(trainer.poses_rot), 0)
                if cfg.local.progressive_opt
                else None
            ),
            poses_t=(
                torch.stack(list(trainer.poses_t), 0)
                if cfg.local.progressive_opt
                else None
            ),
            blending_weights=(
                trainer.blending_weights if cfg.local.progressive_opt else None
            ),
        )


if __name__ == "__main__":
    # Load config file from base config, yaml and cli.
    base_cfg = OmegaConf.structured(Config())
    cli_cfg = OmegaConf.from_cli()
    base_yaml_path = base_cfg.get("config", None)
    yaml_path = cli_cfg.get("config", None)
    if yaml_path is not None:
        yaml_cfg = OmegaConf.load(yaml_path)
    elif base_yaml_path is not None:
        yaml_cfg = OmegaConf.load(base_yaml_path)
    else:
        yaml_cfg = OmegaConf.create()
    cfg = OmegaConf.merge(base_cfg, yaml_cfg, cli_cfg)  # merge configs

    print(f"Experiment name {cfg.expname}")

    # Fix Random Seed for Reproducibility.
    random.seed(cfg.systems.seed)
    np.random.seed(cfg.systems.seed)
    torch.manual_seed(cfg.systems.seed)
    torch.cuda.manual_seed(cfg.systems.seed)

    if cfg.render_only and (cfg.render_test or cfg.render_path):
        # Inference only.
        render_test(cfg)
    elif cfg.local_models:
        reconstruction_local(cfg)
