import os
import wandb
import argparse
import numpy as np
import yaml
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler

# --- Imports ---
from vint_train.models.vint.vint import ViNT
from vint_train.models.vint.depth_vint import DepthViNT

# Fixed: Import ViNTDatasetDepth from the same module as ViNT_Dataset
from vint_train.data.vint_dataset import ViNT_Dataset, ViNTDatasetDepth
from vint_train.training.train_eval_loop import (
    train_eval_loop,
    train_eval_loop_vint_depth,
    load_model,
)


def main(config):
    assert config["distance"]["min_dist_cat"] < config["distance"]["max_dist_cat"]
    assert config["action"]["min_dist_cat"] < config["action"]["max_dist_cat"]

    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["gpu_ids"]]
        )
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    cudnn.benchmark = True
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- Data Loading ---
    train_dataset = []
    test_dataloaders = {}

    if "context_type" not in config:
        config["context_type"] = "temporal"

    if "clip_goals" not in config:
        config["clip_goals"] = False
    
    if "contrastive_weight" not in config:
        config["contrastive_weight"] = 0.1
    
    # Fixed: Add default alpha value
    if "alpha" not in config:
        config["alpha"] = 0.5

    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        if "negative_mining" not in data_config:
            data_config["negative_mining"] = True
        if "goals_per_obs" not in data_config:
            data_config["goals_per_obs"] = 1
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0
        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1

        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                # Choose dataset class based on model type
                if config["model_type"] == "vint_depth":
                    dataset_class = ViNTDatasetDepth
                else:
                    dataset_class = ViNT_Dataset
                    
                dataset = dataset_class(
                    data_folder=data_config["data_folder"],
                    data_split_folder=data_config[data_split_type],
                    dataset_name=dataset_name,
                    image_size=config["image_size"],
                    waypoint_spacing=data_config["waypoint_spacing"],
                    min_dist_cat=config["distance"]["min_dist_cat"],
                    max_dist_cat=config["distance"]["max_dist_cat"],
                    min_action_distance=config["action"]["min_dist_cat"],
                    max_action_distance=config["action"]["max_dist_cat"],
                    negative_mining=data_config["negative_mining"],
                    len_traj_pred=config["len_traj_pred"],
                    learn_angle=config["learn_angle"],
                    context_size=config["context_size"],
                    context_type=config["context_type"],
                    end_slack=data_config["end_slack"],
                    goals_per_obs=data_config["goals_per_obs"],
                    normalize=config["normalize"],
                    goal_type=config["goal_type"],
                )
                
                if data_split_type == "train":
                    train_dataset.append(dataset)
                else:
                    dataset_type = f"{dataset_name}_{data_split_type}"
                    if dataset_type not in test_dataloaders:
                        test_dataloaders[dataset_type] = {}
                    test_dataloaders[dataset_type] = dataset

    train_dataset = ConcatDataset(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=False,
        persistent_workers=True,
    )

    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]

    for dataset_type, dataset in test_dataloaders.items():
        test_dataloaders[dataset_type] = DataLoader(
            dataset,
            batch_size=config["eval_batch_size"],
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

    # --- Model Creation ---
    if config["model_type"] == "vint":
        model = ViNT(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    elif config["model_type"] == "vint_depth":
        model = DepthViNT(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
            freeze_vision_encoder=True,
            use_cross_attention=True,
        )
    else:
        raise ValueError(f"Model {config['model_type']} not supported. Use 'vint' or 'vint_depth'.")

    if config["clipping"]:
        print("Clipping gradients to", config["max_norm"])
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.register_hook(
                lambda grad: torch.clamp(grad, -1 * config["max_norm"], config["max_norm"])
            )

    lr = float(config["lr"])
    config["optimizer"] = config["optimizer"].lower()
    
    # Filter for trainable parameters
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    if config["optimizer"] == "adam":
        optimizer = Adam(trainable_params, lr=lr, betas=(0.9, 0.98))
    elif config["optimizer"] == "adamw":
        optimizer = AdamW(trainable_params, lr=lr)
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")

    scheduler = None
    if config["scheduler"] is not None:
        config["scheduler"] = config["scheduler"].lower()
        if config["scheduler"] == "cosine":
            print("Using cosine annealing with T_max", config["epochs"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
        elif config["scheduler"] == "cyclic":
            print("Using cyclic LR with cycle", config["cyclic_period"])
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=lr / 10.0, max_lr=lr, step_size_up=config["cyclic_period"] // 2, cycle_momentum=False
            )
        elif config["scheduler"] == "plateau":
            print("Using ReduceLROnPlateau")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=config["plateau_factor"], patience=config["plateau_patience"], verbose=True
            )
        else:
            raise ValueError(f"Scheduler {config['scheduler']} not supported")

        if config["warmup"]:
            print("Using warmup scheduler")
            scheduler = GradualWarmupScheduler(
                optimizer, multiplier=1, total_epoch=config["warmup_epochs"], after_scheduler=scheduler
            )

    current_epoch = 0
    checkpoint_to_load = None
    load_optimizer_and_scheduler = False

    # First, check if we are RESUMING a previous run
    if "load_run" in config:
        load_project_folder = os.path.join("logs", config["load_run"])
        resume_path = os.path.join(load_project_folder, "latest.pth")
        if os.path.exists(resume_path):
            print(f"--- Resuming training from checkpoint: {resume_path} ---")
            checkpoint_to_load = torch.load(resume_path)
            # If we are resuming, we load the epoch, optimizer, and scheduler later
            if "epoch" in checkpoint_to_load:
                current_epoch = checkpoint_to_load["epoch"] + 1
            load_optimizer_and_scheduler = True
        else:
            print(f"WARNING: 'load_run' was specified, but checkpoint not found at {resume_path}")

    # If we are NOT resuming, check if we should START fine-tuning from a vint.pth file
    if current_epoch == 0 and "finetune_from_vint_pth" in config:
        pretrained_path = config["finetune_from_vint_pth"]
        if os.path.exists(pretrained_path):
            print(f"--- Starting new fine-tuning run from: {pretrained_path} ---")
            # We only load the model weights, not the epoch or optimizer state
            checkpoint_to_load = torch.load(pretrained_path)
            load_optimizer_and_scheduler = False
        else:
            print(f"WARNING: Pretrained VINT path specified but not found: {pretrained_path}")

    # If a checkpoint was found (either for resuming or fine-tuning), load the model weights
    if checkpoint_to_load is not None:
        load_model(model, config["model_type"], checkpoint_to_load)

    # Move model to device(s)
    if len(config["gpu_ids"]) > 1:
        model = nn.DataParallel(model, device_ids=config["gpu_ids"])
    model = model.to(device)

    # Fixed: Load optimizer and scheduler states only if resuming
    if load_optimizer_and_scheduler and checkpoint_to_load is not None:
        if "optimizer" in checkpoint_to_load:
            optimizer.load_state_dict(checkpoint_to_load["optimizer"])
            print("Loaded optimizer state from checkpoint")
        if scheduler is not None and "scheduler" in checkpoint_to_load:
            scheduler.load_state_dict(checkpoint_to_load["scheduler"])
            print("Loaded scheduler state from checkpoint")

    # --- Training Loop ---
    if config["model_type"] == "vint_depth":
        train_eval_loop_vint_depth(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            normalized=config["normalize"],
            print_log_freq=config["print_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            learn_angle=config["learn_angle"],
            contrastive_weight=config["contrastive_weight"],
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
        )
    else:  # This will handle the "vint" model type
        train_eval_loop(
            train_model=config["train"],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            normalized=config["normalize"],
            print_log_freq=config["print_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            learn_angle=config["learn_angle"],
            alpha=config["alpha"],
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
        )

    print("FINISHED TRAINING")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")

    parser.add_argument(
        "--config",
        "-c",
        default="config/vint.yaml",
        type=str,
        help="Path to the config file",
    )
    args = parser.parse_args()

    with open("config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    config = default_config

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)

    config.update(user_config)

    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]
    )
    os.makedirs(config["project_folder"])

    if config["use_wandb"]:
        wandb.login()
        wandb.init(
            project=config["project_name"],
            settings=wandb.Settings(start_method="fork"),
            entity="your_wandb_entity",  # TODO: change this to your wandb entity
        )
        wandb.save(args.config, policy="now")
        wandb.run.name = config["run_name"]
        if wandb.run:
            wandb.config.update(config)

    print(config)
    main(config)