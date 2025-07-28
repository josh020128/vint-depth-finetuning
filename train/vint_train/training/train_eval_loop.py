# vint_train/training/train_eval_loop.py

import torch
import torch.nn as nn
import wandb
import numpy as np
import os
import tqdm
from prettytable import PrettyTable

from vint_train.training.train_utils import (
    save_model,
)

def train_eval_loop(
    train_model,
    model,
    optimizer,
    scheduler,
    dataloader,
    test_dataloaders,
    transform,
    epochs,
    device,
    project_folder,
    normalized,
    print_log_freq,
    image_log_freq,
    num_images_log,
    current_epoch,
    learn_angle,
    alpha,
    use_wandb,
    eval_fraction,
):
    """
    Main training and evaluation loop for standard ViNT model.
    """
    if use_wandb:
        wandb.watch(model, log="all", log_freq=print_log_freq)

    action_loss_fn = nn.MSELoss()

    for epoch in range(current_epoch, epochs):
        model.train()
        epoch_total_loss = 0

        tqdm_iter = tqdm.tqdm(
            dataloader, 
            disable=not(print_log_freq > 0), 
            dynamic_ncols=True,
            desc=f"Epoch {epoch}/{epochs}"
        )

        for i, data in enumerate(tqdm_iter):
            obs_img, goal_img, _, _, actions, dist_goal, _, _, action_mask = data
            
            obs_img, goal_img, actions, dist_goal, action_mask = (
                obs_img.to(device),
                goal_img.to(device),
                actions.to(device),
                dist_goal.to(device),
                action_mask.to(device),
            )

            optimizer.zero_grad()
            dist_pred, action_pred = model(obs_img, goal_img)

            action_loss = action_loss_fn(action_pred, actions)
            dist_loss = nn.CrossEntropyLoss()(dist_pred, dist_goal)
            
            total_loss = alpha * dist_loss + (1 - alpha) * action_loss
            total_loss.backward()
            optimizer.step()

            epoch_total_loss += total_loss.item()
            
            if use_wandb and i % wandb.run.config.wandb_log_freq == 0:
                wandb.log(
                    {"train/step_loss": total_loss.item(), "train/learning_rate": optimizer.param_groups[0]['lr']},
                    step=epoch * len(dataloader) + i,
                )

        if scheduler is not None:
            scheduler.step()
        
        if use_wandb:
            wandb.log({"train/epoch_loss": epoch_total_loss / len(dataloader), "epoch": epoch}, step=(epoch+1)*len(dataloader))

        # Evaluation
        avg_total_test_loss = []
        if len(test_dataloaders) > 0:
            for dataset_name, test_dataloader in test_dataloaders.items():
                test_dist_loss, test_action_loss, total_eval_loss = evaluate(
                    model, test_dataloader, device, epoch, dataset_name, normalized, learn_angle, alpha, use_wandb, eval_fraction
                )
                avg_total_test_loss.append(total_eval_loss)

        save_model(model, optimizer, scheduler, epoch, project_folder, "latest")
        if (epoch + 1) % 5 == 0:
            save_model(model, optimizer, scheduler, epoch, project_folder, f"epoch_{epoch}")

def evaluate(
    model, dataloader, device, epoch, dataset_name, normalized, learn_angle, alpha, use_wandb, eval_fraction
):
    """Evaluation function for standard ViNT"""
    model.eval()
    total_dist_loss = 0
    total_action_loss = 0
    num_datapoints = 0

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i > len(dataloader) * eval_fraction:
                break
            
            obs_img, goal_img, _, _, actions, dist_goal, _, _, action_mask = data
            
            obs_img, goal_img, actions, dist_goal, action_mask = (
                obs_img.to(device),
                goal_img.to(device),
                actions.to(device),
                dist_goal.to(device),
                action_mask.to(device),
            )

            dist_pred, action_pred = model(obs_img, goal_img)

            action_loss = nn.MSELoss()(action_pred, actions)
            dist_loss = nn.CrossEntropyLoss()(dist_pred, dist_goal)

            total_dist_loss += dist_loss.item() * obs_img.shape[0]
            total_action_loss += action_loss.item() * obs_img.shape[0]
            num_datapoints += obs_img.shape[0]

    avg_dist_loss = total_dist_loss / num_datapoints
    avg_action_loss = total_action_loss / num_datapoints
    total_eval_loss = alpha * avg_dist_loss + (1 - alpha) * avg_action_loss

    if use_wandb:
        wandb.log(
            {
                f"eval_{dataset_name}/epoch_dist_loss": avg_dist_loss,
                f"eval_{dataset_name}/epoch_action_loss": avg_action_loss,
                f"eval_{dataset_name}/epoch_total_loss": total_eval_loss,
                "epoch": epoch,
            }
        )
    print(f"Eval {dataset_name}: Epoch {epoch}, Dist Loss: {avg_dist_loss:.4f}, Action Loss: {avg_action_loss:.4f}, Total Loss: {total_eval_loss:.4f}")
    return avg_dist_loss, avg_action_loss, total_eval_loss


def train_eval_loop_vint_depth(
    model,
    optimizer,
    scheduler,
    dataloader,
    test_dataloaders,
    transform,
    epochs,
    device,
    project_folder,
    normalized,
    print_log_freq,
    image_log_freq,
    num_images_log,
    current_epoch,
    learn_angle,
    contrastive_weight,
    use_wandb,
    eval_fraction,
):
    """
    Main training and evaluation loop for DepthViNT model.
    """
    if use_wandb:
        wandb.watch(model, log="all", log_freq=print_log_freq)

    for epoch in range(current_epoch, epochs):
        model.train()
        
        epoch_total_loss = 0
        epoch_action_loss = 0
        epoch_contrastive_loss = 0
        
        tqdm_iter = tqdm.tqdm(
            dataloader, 
            disable=not(print_log_freq > 0), 
            dynamic_ncols=True,
            desc=f"Epoch {epoch}/{epochs}"
        )

        for i, data in enumerate(tqdm_iter):
            obs_img, goal_img, obs_depth, goal_depth, actions, _, _, _, action_mask = data
            
            obs_img, goal_img, obs_depth, goal_depth, actions, action_mask = (
                obs_img.to(device),
                goal_img.to(device),
                obs_depth.to(device),
                goal_depth.to(device),
                actions.to(device),
                action_mask.to(device),
            )

            optimizer.zero_grad()
            dist_pred, action_pred, vision_features, depth_features = model(
                obs_img, goal_img, obs_depth, goal_depth
            )

            action_loss = torch.mean((action_pred - actions) ** 2 * action_mask.unsqueeze(-1))
            
            contrastive_loss = torch.tensor(0.0, device=device)
            if vision_features is not None and depth_features is not None:
                contrastive_loss_fn = model.module.contrastive_loss_fn if isinstance(model, nn.DataParallel) else model.contrastive_loss_fn
                contrastive_loss = contrastive_loss_fn(vision_features, depth_features)
            
            total_loss = action_loss + contrastive_weight * contrastive_loss
            
            total_loss.backward()
            optimizer.step()

            epoch_total_loss += total_loss.item()
            epoch_action_loss += action_loss.item()
            epoch_contrastive_loss += contrastive_loss.item()

            if print_log_freq > 0 and i % print_log_freq == 0:
                tqdm_iter.set_postfix({
                    "loss": total_loss.item(),
                    "action_loss": action_loss.item(),
                    "contr_loss": contrastive_loss.item(),
                })

            if use_wandb and i % wandb.run.config.wandb_log_freq == 0:
                wandb.log(
                    {
                        "train/step_loss": total_loss.item(),
                        "train/step_action_loss": action_loss.item(),
                        "train/step_contrastive_loss": contrastive_loss.item(),
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                    },
                    step=epoch * len(dataloader) + i,
                )

        if scheduler is not None:
            scheduler.step()

        if use_wandb:
            wandb.log(
                {
                    "train/epoch_loss": epoch_total_loss / len(dataloader),
                    "train/epoch_action_loss": epoch_action_loss / len(dataloader),
                    "train/epoch_contrastive_loss": epoch_contrastive_loss / len(dataloader),
                    "epoch": epoch,
                },
                step=(epoch + 1) * len(dataloader),
            )
            
        if len(test_dataloaders) > 0:
            for dataset_name, test_dataloader in test_dataloaders.items():
                evaluate_vint_depth(
                    model=model,
                    dataloader=test_dataloader,
                    device=device,
                    epoch=epoch,
                    dataset_name=dataset_name,
                    contrastive_weight=contrastive_weight,
                    use_wandb=use_wandb,
                    eval_fraction=eval_fraction,
                )

        save_model(model, optimizer, scheduler, epoch, project_folder, "latest")
        if (epoch + 1) % 5 == 0:
            save_model(model, optimizer, scheduler, epoch, project_folder, f"epoch_{epoch}")


def evaluate_vint_depth(
    model,
    dataloader,
    device,
    epoch,
    dataset_name,
    contrastive_weight,
    use_wandb,
    eval_fraction,
):
    """
    Evaluation loop for DepthViNT model.
    """
    model.eval()
    
    total_action_loss = 0
    total_contrastive_loss = 0
    num_datapoints = 0

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i > len(dataloader) * eval_fraction:
                break
            
            obs_img, goal_img, obs_depth, goal_depth, actions, _, _, _, action_mask = data
            
            obs_img, goal_img, obs_depth, goal_depth, actions, action_mask = (
                obs_img.to(device),
                goal_img.to(device),
                obs_depth.to(device),
                goal_depth.to(device),
                actions.to(device),
                action_mask.to(device),
            )

            dist_pred, action_pred, vision_features, depth_features = model(
                obs_img, goal_img, obs_depth, goal_depth
            )

            action_loss = torch.mean((action_pred - actions) ** 2 * action_mask.unsqueeze(-1))
            
            contrastive_loss = torch.tensor(0.0, device=device)
            if vision_features is not None and depth_features is not None:
                contrastive_loss_fn = model.module.contrastive_loss_fn if isinstance(model, nn.DataParallel) else model.contrastive_loss_fn
                contrastive_loss = contrastive_loss_fn(vision_features, depth_features)
            
            total_action_loss += action_loss.item() * obs_img.shape[0]
            total_contrastive_loss += contrastive_loss.item() * obs_img.shape[0]
            num_datapoints += obs_img.shape[0]

    avg_action_loss = total_action_loss / num_datapoints
    avg_contrastive_loss = total_contrastive_loss / num_datapoints
    avg_total_loss = avg_action_loss + contrastive_weight * avg_contrastive_loss

    if use_wandb:
        wandb.log(
            {
                f"eval_{dataset_name}/epoch_loss": avg_total_loss,
                f"eval_{dataset_name}/epoch_action_loss": avg_action_loss,
                f"eval_{dataset_name}/epoch_contrastive_loss": avg_contrastive_loss,
                "epoch": epoch,
            }
        )
    
    print(f"Eval {dataset_name}: Epoch {epoch}, Loss: {avg_total_loss:.4f}, Action Loss: {avg_action_loss:.4f}, Contrastive Loss: {avg_contrastive_loss:.4f}")


def load_model(model, model_type, checkpoint: dict) -> None:
    """Load model from checkpoint."""
    loaded_model = checkpoint["model"]
    try:
        state_dict = loaded_model.module.state_dict()
        model.load_state_dict(state_dict, strict=False)
    except AttributeError:
        state_dict = loaded_model.state_dict()
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