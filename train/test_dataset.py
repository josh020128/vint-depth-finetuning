import torch
import numpy as np
import matplotlib.pyplot as plt
from vint_train.data.vint_dataset import ViNT_Dataset, ViNTDatasetDepth
import os

# Test script to verify SCAND dataset loading
def test_scand_loading():
    # Configure dataset
    config = {
        "data_folder": "/home/airlab/vint_fuse2/SCAND",
        "data_split_folder": "/home/airlab/vint_fuse2/SCAND/train/",
        "dataset_name": "scand",
        "image_size": [85, 64],
        "waypoint_spacing": 1,
        "min_dist_cat": 0,
        "max_dist_cat": 20,
        "min_action_distance": 0,
        "max_action_distance": 10,
        "negative_mining": True,
        "len_traj_pred": 5,
        "learn_angle": True,
        "context_size": 5,
        "context_type": "temporal",
        "end_slack": 0,
        "goals_per_obs": 1,
        "normalize": True,
        "goal_type": "image",
    }
    
    print("Loading SCAND dataset...")
    dataset = ViNTDatasetDepth(**config)
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a few samples
    for i in range(min(3, len(dataset))):
        print(f"\nTesting sample {i}...")
        try:
            data = dataset[i]
            obs_img, goal_img, obs_depth, goal_depth, actions, dist_goal, goal_pos, dataset_index, action_mask = data
            
            print(f"  obs_img shape: {obs_img.shape}")
            print(f"  goal_img shape: {goal_img.shape}")
            print(f"  obs_depth shape: {obs_depth.shape}")
            print(f"  goal_depth shape: {goal_depth.shape}")
            print(f"  obs_img range: [{obs_img.min():.3f}, {obs_img.max():.3f}]")
            print(f"  obs_depth range: [{obs_depth.min():.3f}, {obs_depth.max():.3f}]")
            print(f"  action_mask: {action_mask}")
            
            # Visualize the data
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            
            # Show RGB observations (last 3 frames)
            for j in range(3):
                rgb_frame = obs_img[j*3:(j+1)*3].numpy().transpose(1, 2, 0)
                # Denormalize
                rgb_frame = rgb_frame * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                rgb_frame = np.clip(rgb_frame, 0, 1)
                axes[0, j].imshow(rgb_frame)
                axes[0, j].set_title(f'RGB Obs {j}')
                axes[0, j].axis('off')
            
            # Show goal RGB
            goal_rgb = goal_img.numpy().transpose(1, 2, 0)
            goal_rgb = goal_rgb * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            goal_rgb = np.clip(goal_rgb, 0, 1)
            axes[0, 3].imshow(goal_rgb)
            axes[0, 3].set_title('Goal RGB')
            axes[0, 3].axis('off')
            
            # Show depth observations (last 3 frames)
            for j in range(3):
                if j < obs_depth.shape[0]:
                    depth_frame = obs_depth[j].numpy()
                    axes[1, j].imshow(depth_frame, cmap='viridis')
                    axes[1, j].set_title(f'Depth Obs {j}')
                    axes[1, j].axis('off')
            
            # Show goal depth
            goal_depth_img = goal_depth[0].numpy()
            axes[1, 3].imshow(goal_depth_img, cmap='viridis')
            axes[1, 3].set_title('Goal Depth')
            axes[1, 3].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'scand_test_sample_{i}.png')
            plt.close()
            
            print(f"  Saved visualization to scand_test_sample_{i}.png")
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Check if depth images exist
    print("\nChecking for depth images in SCAND dataset...")
    traj_names_file = os.path.join(config["data_split_folder"], "traj_names.txt")
    with open(traj_names_file, "r") as f:
        traj_names = f.read().strip().split("\n")
    
    for traj_name in traj_names[:3]:  # Check first 3 trajectories
        traj_path = os.path.join(config["data_folder"], traj_name)
        print(f"\nTrajectory: {traj_name}")
        
        # List files in trajectory
        if os.path.exists(traj_path):
            files = os.listdir(traj_path)
            rgb_files = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]
            depth_files = [f for f in files if 'depth' in f]
            
            print(f"  RGB files: {len(rgb_files)} (example: {rgb_files[0] if rgb_files else 'None'})")
            print(f"  Depth files: {len(depth_files)} (example: {depth_files[0] if depth_files else 'None'})")

if __name__ == "__main__":
    test_scand_loading()