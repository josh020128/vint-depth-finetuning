import torch.nn.functional as F

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

            # FIXED: Correct action loss calculation
            action_loss_unreduced = F.mse_loss(action_pred, actions, reduction='none').mean(dim=(1, 2))  # Shape (B,)
            action_loss = (action_loss_unreduced * action_mask).sum() / (action_mask.sum() + 1e-8)
            
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

        # Rest of the function remains the same...


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

            # FIXED: Correct action loss calculation
            action_loss_unreduced = F.mse_loss(action_pred, actions, reduction='none').mean(dim=(1, 2))  # Shape (B,)
            action_loss = (action_loss_unreduced * action_mask).sum() / (action_mask.sum() + 1e-8)
            
            contrastive_loss = torch.tensor(0.0, device=device)
            if vision_features is not None and depth_features is not None:
                contrastive_loss_fn = model.module.contrastive_loss_fn if isinstance(model, nn.DataParallel) else model.contrastive_loss_fn
                contrastive_loss = contrastive_loss_fn(vision_features, depth_features)
            
            # For evaluation, we accumulate the sum of losses
            valid_samples = action_mask.sum().item()
            total_action_loss += action_loss.item() * valid_samples
            total_contrastive_loss += contrastive_loss.item() * valid_samples
            num_datapoints += valid_samples

    # Average over valid datapoints
    if num_datapoints > 0:
        avg_action_loss = total_action_loss / num_datapoints
        avg_contrastive_loss = total_contrastive_loss / num_datapoints
    else:
        avg_action_loss = 0.0
        avg_contrastive_loss = 0.0
        
    avg_total_loss = avg_action_loss + contrastive_weight * avg_contrastive_loss

    # Rest of the function remains the same...


# DELETE the entire load_model function from this file:
# def load_model(model, model_type, checkpoint: dict) -> None:
#     """Load model from checkpoint."""
#     loaded_model = checkpoint["model"]
#     try:
#         state_dict = loaded_model.module.state_dict()
#         model.load_state_dict(state_dict, strict=False)
#     except AttributeError:
#         state_dict = loaded_model.state_dict()
#         model.load_state_dict(state_dict, strict=False)