import torch
import torch.nn as nn
import torch.nn.functional as F
from vint import ViNT 
from vint_train.models.vint.modules.depth_encoders import DepthEncoder
from vint_train.models.vint.modules.fuse_modules import MultimodalContrastiveLoss, CrossModalAttention

class DepthViNT(ViNT):
    def __init__(self,
                 # ViNT parameters
                 context_size=5,
                 len_traj_pred=5,
                 learn_angle=True,
                 obs_encoder="efficientnet-b0",
                 obs_encoding_size=512,
                 late_fusion=False,
                 mha_num_attention_heads=2,
                 mha_num_attention_layers=2,
                 mha_ff_dim_factor=4,
                 # Depth-specific parameters
                 depth_encoder_name="efficientnet-b0",
                 freeze_vision_encoder=True,
                 use_cross_attention=True):

        # Force late_fusion=True for DepthViNT since we handle fusion differently
        # This ensures goal_encoder expects 3 channels, not 6
        super().__init__(
            context_size, len_traj_pred, learn_angle,
            obs_encoder, obs_encoding_size, True,  # Always use late_fusion=True
            mha_num_attention_heads, mha_num_attention_layers,
            mha_ff_dim_factor
        )

        # Add depth components
        self.depth_encoder = DepthEncoder(depth_encoder_name, obs_encoding_size)
        self.depth_goal_encoder = DepthEncoder(depth_encoder_name, obs_encoding_size)

        # Cross-modal fusion module
        if use_cross_attention:
            self.vision_depth_fusion = CrossModalAttention(obs_encoding_size, num_heads=mha_num_attention_heads)
        else:
            # Simple concatenation + projection
            self.vision_depth_fusion = nn.Sequential(
                nn.Linear(obs_encoding_size * 2, obs_encoding_size),
                nn.ReLU(),
            )
        self.use_cross_attention = use_cross_attention

        # FuSe contrastive loss function
        self.contrastive_loss_fn = MultimodalContrastiveLoss()

        # Freezing strategy (following FuSe)
        if freeze_vision_encoder:
            for param in self.obs_encoder.parameters():
                param.requires_grad = False
            for param in self.goal_encoder.parameters():
                param.requires_grad = False

    def forward(self, obs_img, goal_img, obs_depth=None, goal_depth=None):
        """
        Args:
            obs_img: RGB observations [B, 3 * context_size, H, W] (note: might include current frame)
            goal_img: RGB goal [B, 3, H, W]
            obs_depth: Depth observations [B, context_size, H, W] or [B, context_size + 1, H, W]
            goal_depth: Depth goal [B, 1, H, W]
        """
        batch_size = obs_img.shape[0]
        
        # Determine number of RGB frames from input
        num_rgb_frames = obs_img.shape[1] // 3
        
        # Check if we have context_size or context_size + 1 frames
        if num_rgb_frames == self.context_size:
            # We need to add the current frame to match parent's expectation
            # The parent ViNT expects context_size + 1 frames total
            print(f"Warning: Expected {self.context_size + 1} RGB frames, got {num_rgb_frames}. This might cause issues.")
        
        # 1. Get RGB features
        vision_obs_enc = self.encode_obs_stack(obs_img, self.obs_encoder, self.compress_obs_enc, num_channels=3)
        vision_goal_enc = self.encode_single(goal_img, self.goal_encoder, self.compress_goal_enc)

        # 2. If depth is provided, get depth features and fuse
        if obs_depth is not None and goal_depth is not None:
            # Adjust depth to match RGB frame count if needed
            if obs_depth.shape[1] != num_rgb_frames:
                print(f"Depth frames ({obs_depth.shape[1]}) don't match RGB frames ({num_rgb_frames})")
                # Pad or trim depth to match
                if obs_depth.shape[1] < num_rgb_frames:
                    # Pad with zeros
                    padding = torch.zeros(batch_size, num_rgb_frames - obs_depth.shape[1], 
                                        obs_depth.shape[2], obs_depth.shape[3], device=obs_depth.device)
                    obs_depth = torch.cat([obs_depth, padding], dim=1)
                else:
                    # Trim
                    obs_depth = obs_depth[:, :num_rgb_frames]
                    
            depth_obs_enc = self.encode_obs_stack(obs_depth, self.depth_encoder, nn.Identity(), num_channels=1)
            depth_goal_enc = self.encode_single(goal_depth, self.depth_goal_encoder, nn.Identity())

            # Fuse observation and goal features
            if self.use_cross_attention:
                # Fuse observation context (B, num_frames, D)
                obs_enc = self.vision_depth_fusion(vision_obs_enc, depth_obs_enc)
                # Fuse goal (B, D) -> (B, 1, D) for attention
                goal_enc = self.vision_depth_fusion(
                    vision_goal_enc.unsqueeze(1),
                    depth_goal_enc.unsqueeze(1)
                ).squeeze(1)
            else: # Concat + Linear
                obs_combined = torch.cat([vision_obs_enc, depth_obs_enc], dim=-1)
                obs_enc = self.vision_depth_fusion(obs_combined)

                goal_combined = torch.cat([vision_goal_enc, depth_goal_enc], dim=-1)
                goal_enc = self.vision_depth_fusion(goal_combined)

            # Store features from the current timestep for contrastive loss
            last_vision_features = vision_obs_enc[:, -1, :]
            last_depth_features = depth_obs_enc[:, -1, :]
            
        else: # No depth data, use vision features only
            obs_enc = vision_obs_enc
            goal_enc = vision_goal_enc
            last_vision_features, last_depth_features = None, None

        # 3. Reshape and pass to transformer decoder
        # Ensure goal_enc has the right shape
        if len(goal_enc.shape) == 2:
            goal_enc = goal_enc.unsqueeze(1)
            
        # The parent's decoder expects exactly context_size + 2 tokens
        # Current obs_enc might have context_size or context_size + 1 frames
        current_seq_len = obs_enc.shape[1]
        expected_seq_len = self.context_size + 2
        
        if current_seq_len + 1 == expected_seq_len:
            # We have the right number for obs, just add goal
            tokens = torch.cat((obs_enc, goal_enc), dim=1)
        elif current_seq_len == expected_seq_len - 2:
            # We're missing the "current" frame, duplicate the last one
            current_frame = obs_enc[:, -1:, :]
            tokens = torch.cat((obs_enc, current_frame, goal_enc), dim=1)
        else:
            raise ValueError(f"Unexpected sequence length. Got {current_seq_len} observations, "
                           f"expected {expected_seq_len - 1} or {expected_seq_len - 2}")
        
        final_repr = self.decoder(tokens)

        # 4. Predict distance and actions
        dist_pred = self.dist_predictor(final_repr)
        action_pred = self.action_predictor(final_repr)

        # 5. Format actions
        action_pred = self.format_actions(action_pred)

        # Return predictions and features for contrastive loss
        return dist_pred, action_pred, last_vision_features, last_depth_features

    def encode_obs_stack(self, obs_stack, encoder, compressor, num_channels):
        """Helper to encode a stack of observations."""
        batch_size = obs_stack.shape[0]
        
        # The parent ViNT expects context_size + 1 observations (including current frame)
        # So we need to handle this correctly
        num_frames = self.context_size + 1
        
        # Split the context stack into individual images
        obs_list = torch.split(obs_stack, num_channels, dim=1)
        
        # Ensure we have the right number of frames
        assert len(obs_list) == num_frames, f"Expected {num_frames} frames, got {len(obs_list)}"
        
        obs_cat = torch.cat(obs_list, dim=0)

        # Pass through encoder
        if hasattr(encoder, 'extract_features'):
            obs_enc = encoder.extract_features(obs_cat)
            obs_enc = encoder._avg_pooling(obs_enc)
            if encoder._global_params.include_top:
                obs_enc = obs_enc.flatten(start_dim=1)
                obs_enc = encoder._dropout(obs_enc)
        else:
            # For custom depth encoder
            obs_enc = encoder(obs_cat)
        
        # Apply projection/compression
        obs_enc = compressor(obs_enc)
        
        # Reshape back to (B, num_frames, D)
        obs_enc = obs_enc.reshape((batch_size, num_frames, -1))
        return obs_enc

    def encode_single(self, img, encoder, compressor):
        """Helper to encode a single image (goal)."""
        if hasattr(encoder, 'extract_features'):
            enc = encoder.extract_features(img)
            enc = encoder._avg_pooling(enc)
            if encoder._global_params.include_top:
                enc = enc.flatten(start_dim=1)
                enc = encoder._dropout(enc)
        else:
            # For custom depth encoder
            enc = encoder(img)
        
        # Apply projection/compression
        enc = compressor(enc)
        return enc

    def format_actions(self, action_pred):
        """Helper to format action predictions."""
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(action_pred[:, :, :2], dim=1)
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(action_pred[:, :, 2:].clone(), dim=-1)
        return action_pred