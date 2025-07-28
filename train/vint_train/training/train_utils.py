import wandb
import os
import numpy as np
import yaml
from typing import List, Optional, Dict
from prettytable import PrettyTable
import tqdm
import itertools

from vint_train.visualizing.action_utils import visualize_traj_pred
from vint_train.visualizing.distance_utils import visualize_dist_pred
from vint_train.visualizing.visualize_utils import to_numpy
from vint_train.training.logger import Logger
from vint_train.data.data_utils import VISUALIZATION_IMAGE_SIZE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
import torchvision.transforms.functional as TF

# =================================================================================
# Train utils for Standard ViNT Model
# =================================================================================

def _compute_losses(
    dist_label: torch.Tensor,
    action_label: torch.Tensor,
    dist_pred: torch.Tensor,
    action_pred: torch.Tensor,
    alpha: float,
    learn_angle: bool,
    action_mask: torch.Tensor = None,
):
    """
    Compute losses for distance and action prediction.
    """
    dist_loss = F.mse_loss(dist_pred.squeeze(-1), dist_label.float())

    def action_reduce(unreduced_loss: torch.Tensor):
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    assert action_pred.shape == action_label.shape, f"{action_pred.shape} != {action_label.shape}"
    action_loss = action_reduce(F.mse_loss(action_pred, action_label, reduction="none"))

    results = {"dist_loss": dist_loss, "action_loss": action_loss}
    total_loss = alpha * 1e-2 * dist_loss + (1 - alpha) * action_loss
    results["total_loss"] = total_loss

    return results


def _log_data(
    i, epoch, num_batches, normalized, project_folder, num_images_log, loggers,
    obs_image, goal_image, action_pred, action_label, dist_pred, dist_label,
    goal_pos, dataset_index, use_wandb, mode, use_latest,
    wandb_log_freq=1, print_log_freq=1, image_log_freq=1, wandb_increment_step=True,
):
    """
    Log data to wandb and print to console.
    """
    data_log = {}
    for key, logger in loggers.items():
        if use_latest:
            data_log[logger.full_name()] = logger.latest()
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")
        else:
            data_log[logger.full_name()] = logger.average()
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(f"(epoch {epoch}) {logger.full_name()} {logger.average()}")

    if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
        wandb.log(data_log, commit=wandb_increment_step)

    if image_log_freq != 0 and i % image_log_freq == 0:
        visualize_dist_pred(
            to_numpy(obs_image), to_numpy(goal_image), to_numpy(dist_pred),
            to_numpy(dist_label), mode, project_folder, epoch,
            num_images_log, use_wandb=use_wandb,
        )
        visualize_traj_pred(
            to_numpy(obs_image), to_numpy(goal_image), to_numpy(dataset_index),
            to_numpy(goal_pos), to_numpy(action_pred), to_numpy(action_label),
            mode, normalized, project_folder, epoch, num_images_log, use_wandb=use_wandb,
        )


def train(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    epoch: int,
    alpha: float = 0.5,
    learn_angle: bool = True,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    use_tqdm: bool = True,
):
    """
    Train the model for one epoch.
    """
    model.train()
    dist_loss_logger = Logger("dist_loss", "train", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)
    loggers = {
        "dist_loss": dist_loss_logger,
        "action_loss": action_loss_logger,
        "total_loss": total_loss_logger,
    }

    num_batches = len(dataloader)
    tqdm_iter = tqdm.tqdm(dataloader, disable=not use_tqdm, dynamic_ncols=True, desc=f"Training epoch {epoch}")
    for i, data in enumerate(tqdm_iter):
        (
            obs_image, goal_image, action_label, dist_label, goal_pos, dataset_index, action_mask,
        ) = data

        obs_images = torch.split(obs_image, 3, dim=1)
        viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)
        obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
        obs_image = torch.cat(obs_images, dim=1)

        viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE)
        
        goal_image = transform(goal_image).to(device)
        dist_pred, action_pred = model(obs_image, goal_image)

        dist_label, action_label, action_mask = dist_label.to(device), action_label.to(device), action_mask.to(device)

        optimizer.zero_grad()
        
        losses = _compute_losses(
            dist_label=dist_label, action_label=action_label, dist_pred=dist_pred,
            action_pred=action_pred, alpha=alpha, learn_angle=learn_angle, action_mask=action_mask,
        )

        losses["total_loss"].backward()
        optimizer.step()

        for key, value in losses.items():
            if key in loggers:
                loggers[key].log_data(value.item())

        _log_data(
            i=i, epoch=epoch, num_batches=num_batches, normalized=normalized,
            project_folder=project_folder, num_images_log=num_images_log,
            loggers=loggers, obs_image=viz_obs_image, goal_image=viz_goal_image,
            action_pred=action_pred, action_label=action_label, dist_pred=dist_pred,
            dist_label=dist_label, goal_pos=goal_pos, dataset_index=dataset_index,
            wandb_log_freq=wandb_log_freq, print_log_freq=print_log_freq,
            image_log_freq=image_log_freq, use_wandb=use_wandb, mode="train", use_latest=True,
        )


def evaluate(
    eval_type: str,
    model: nn.Module,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    epoch: int = 0,
    alpha: float = 0.5,
    learn_angle: bool = True,
    num_images_log: int = 8,
    use_wandb: bool = True,
    eval_fraction: float = 1.0,
    use_tqdm: bool = True,
):
    """
    Evaluate the model on the given evaluation dataset.
    """
    model.eval()
    dist_loss_logger = Logger("dist_loss", eval_type)
    action_loss_logger = Logger("action_loss", eval_type)
    total_loss_logger = Logger("total_loss", eval_type)
    loggers = {
        "dist_loss": dist_loss_logger, 
        "action_loss": action_loss_logger,
        "total_loss": total_loss_logger,
    }

    num_batches = max(int(len(dataloader) * eval_fraction), 1)

    viz_obs_image = None
    with torch.no_grad():
        tqdm_iter = tqdm.tqdm(itertools.islice(dataloader, num_batches), total=num_batches, disable=not use_tqdm, dynamic_ncols=True, desc=f"Evaluating {eval_type} for epoch {epoch}")
        for i, data in enumerate(tqdm_iter):
            (
                obs_image, goal_image, action_label, dist_label, goal_pos, dataset_index, action_mask,
            ) = data

            obs_images = torch.split(obs_image, 3, dim=1)
            viz_obs_image = TF.resize(obs_images[-1], VISUALIZATION_IMAGE_SIZE)
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image = torch.cat(obs_images, dim=1)

            viz_goal_image = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE)

            goal_image = transform(goal_image).to(device)
            dist_pred, action_pred = model(obs_image, goal_image)

            dist_label, action_label, action_mask = dist_label.to(device), action_label.to(device), action_mask.to(device)

            losses = _compute_losses(
                dist_label=dist_label, action_label=action_label, dist_pred=dist_pred,
                action_pred=action_pred, alpha=alpha, learn_angle=learn_angle, action_mask=action_mask,
            )

            for key, value in losses.items():
                if key in loggers:
                    loggers[key].log_data(value.item())

    _log_data(
        i=i, epoch=epoch, num_batches=num_batches, normalized=normalized,
        project_folder=project_folder, num_images_log=num_images_log, loggers=loggers,
        obs_image=viz_obs_image, goal_image=viz_goal_image, action_pred=action_pred,
        action_label=action_label, goal_pos=goal_pos, dist_pred=dist_pred,
        dist_label=dist_label, dataset_index=dataset_index, use_wandb=use_wandb,
        mode=eval_type, use_latest=False, wandb_increment_step=False,
    )

    return dist_loss_logger.average(), action_loss_logger.average(), total_loss_logger.average()

# =================================================================================
# New Train utils for DepthViNT
# =================================================================================

def _compute_losses_depth(
    action_label: torch.Tensor,
    action_pred: torch.Tensor,
    vision_features: torch.Tensor,
    depth_features: torch.Tensor,
    contrastive_loss_fn: nn.Module,
    contrastive_weight: float,
    action_mask: torch.Tensor,
):
    """
    Compute losses for DepthViNT: action loss and contrastive loss.
    """
    action_loss = torch.mean((action_pred - action_label)**2 * action_mask.unsqueeze(-1))
    
    contrastive_loss = torch.tensor(0.0, device=action_pred.device)
    if vision_features is not None and depth_features is not None:
        contrastive_loss = contrastive_loss_fn(vision_features, depth_features)
    
    total_loss = action_loss + contrastive_weight * contrastive_loss
    
    return {
        "action_loss": action_loss,
        "contrastive_loss": contrastive_loss,
        "total_loss": total_loss,
    }

def _log_data_depth(
    i, epoch, num_batches, loggers, use_wandb, mode, use_latest,
    wandb_log_freq=1, print_log_freq=1,
):
    """
    Log data for DepthViNT.
    """
    data_log = {}
    for key, logger in loggers.items():
        if use_latest:
            data_log[logger.full_name()] = logger.latest()
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")
        else:
            data_log[logger.full_name()] = logger.average()
            if i % print_log_freq == 0 and print_log_freq != 0:
                print(f"(epoch {epoch}) {logger.full_name()} {logger.average()}")

    if use_wandb and i % wandb_log_freq == 0 and wandb_log_freq != 0:
        wandb.log(data_log, commit=True)

def train_vint_depth(
    model: nn.Module,
    optimizer: Adam,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    contrastive_weight: float,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    use_wandb: bool = True,
):
    """
    Train the DepthViNT model for one epoch.
    """
    model.train()
    action_loss_logger = Logger("action_loss", "train_depth", window_size=print_log_freq)
    contrastive_loss_logger = Logger("contrastive_loss", "train_depth", window_size=print_log_freq)
    total_loss_logger = Logger("total_loss", "train_depth", window_size=print_log_freq)
    loggers = {
        "action_loss": action_loss_logger,
        "contrastive_loss": contrastive_loss_logger,
        "total_loss": total_loss_logger,
    }

    num_batches = len(dataloader)
    tqdm_iter = tqdm.tqdm(dataloader, disable=not(print_log_freq > 0), dynamic_ncols=True, desc=f"Training DepthViNT Epoch {epoch}")

    for i, data in enumerate(tqdm_iter):
        obs_img, goal_img, obs_depth, goal_depth, actions, _, _, _, action_mask = data
        
        obs_img, goal_img, obs_depth, goal_depth, actions, action_mask = (
            obs_img.to(device), goal_img.to(device), obs_depth.to(device),
            goal_depth.to(device), actions.to(device), action_mask.to(device),
        )

        optimizer.zero_grad()

        _, action_pred, vision_features, depth_features = model(
            obs_img, goal_img, obs_depth, goal_depth
        )
        
        contrastive_loss_fn = model.module.contrastive_loss_fn if isinstance(model, nn.DataParallel) else model.contrastive_loss_fn
        losses = _compute_losses_depth(
            actions, action_pred, vision_features, depth_features,
            contrastive_loss_fn, contrastive_weight, action_mask
        )

        losses["total_loss"].backward()
        optimizer.step()

        for key, value in losses.items():
            loggers[key].log_data(value.item())

        _log_data_depth(i, epoch, num_batches, loggers, use_wandb, "train_depth", True, wandb_log_freq, print_log_freq)

def evaluate_vint_depth(
    eval_type: str,
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    contrastive_weight: float,
    use_wandb: bool = True,
    eval_fraction: float = 1.0,
):
    """
    Evaluate the DepthViNT model.
    """
    model.eval()
    action_loss_logger = Logger("action_loss", f"eval_{eval_type}")
    contrastive_loss_logger = Logger("contrastive_loss", f"eval_{eval_type}")
    total_loss_logger = Logger("total_loss", f"eval_{eval_type}")
    loggers = {
        "action_loss": action_loss_logger,
        "contrastive_loss": contrastive_loss_logger,
        "total_loss": total_loss_logger,
    }

    num_batches = max(int(len(dataloader) * eval_fraction), 1)
    
    with torch.no_grad():
        tqdm_iter = tqdm.tqdm(itertools.islice(dataloader, num_batches), total=num_batches, dynamic_ncols=True, desc=f"Evaluating {eval_type} for epoch {epoch}")
        for i, data in enumerate(tqdm_iter):
            obs_img, goal_img, obs_depth, goal_depth, actions, _, _, _, action_mask = data
            
            obs_img, goal_img, obs_depth, goal_depth, actions, action_mask = (
                obs_img.to(device), goal_img.to(device), obs_depth.to(device),
                goal_depth.to(device), actions.to(device), action_mask.to(device),
            )

            _, action_pred, vision_features, depth_features = model(
                obs_img, goal_img, obs_depth, goal_depth
            )

            contrastive_loss_fn = model.module.contrastive_loss_fn if isinstance(model, nn.DataParallel) else model.contrastive_loss_fn
            losses = _compute_losses_depth(
                actions, action_pred, vision_features, depth_features,
                contrastive_loss_fn, contrastive_weight, action_mask
            )

            for key, value in losses.items():
                loggers[key].log_data(value.item())

    _log_data_depth(i, epoch, num_batches, loggers, use_wandb, f"eval_{eval_type}", False)

    return total_loss_logger.average()

# =================================================================================
# Functions for Saving and Loading Models
# =================================================================================

def save_model(model, optimizer, scheduler, epoch, project_folder, name):
    """
    Save the model, optimizer, and scheduler state to a file.
    """
    if not os.path.isdir(project_folder):
        os.makedirs(project_folder)
        
    checkpoint_path = os.path.join(project_folder, f"{name}.pth")
    
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    
    checkpoint = {
        "epoch": epoch,
        "model": model_state,
        "optimizer": optimizer.state_dict(),
    }
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
        
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def load_model(model, model_type, checkpoint: dict) -> None:
    """Load model from checkpoint."""
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    if all(k.startswith('module.') for k in state_dict.keys()):
        if not isinstance(model, nn.DataParallel):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
    elif not all(k.startswith('module.') for k in state_dict.keys()) and isinstance(model, nn.DataParallel):
        state_dict = {'module.' + k: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(f"Total Trainable Params: {total_params/1e6:.2f}M")
    return total_params