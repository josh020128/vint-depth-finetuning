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

        # Initialize parent ViNT
        super().__init__(
            context_size, len_traj_pred, learn_angle,
            obs_encoder, obs_encoding_size, late_fusion,
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
            obs_img: RGB observations [B, 3 * context_size, H, W]
            goal_img: RGB goal [B, 3, H, W]
            obs_depth: Depth observations [B, context_size, H, W]
            goal_depth: Depth goal [B, 1, H, W]
        """
        batch_size = obs_img.shape[0]

        # 1. Get RGB features (reusing parent's encoding logic)
        vision_obs_enc = self.encode_obs_stack(obs_img, self.obs_encoder, self.compress_obs_enc, num_channels=3)
        vision_goal_enc = self.encode_single(goal_img, self.goal_encoder, self.compress_goal_enc)

        # 2. If depth is provided, get depth features and fuse
        if obs_depth is not None and goal_depth is not None:
            depth_obs_enc = self.encode_obs_stack(obs_depth, self.depth_encoder, nn.Identity(), num_channels=1)
            depth_goal_enc = self.encode_single(goal_depth, self.depth_goal_encoder, nn.Identity())
            
            # Reshape for fusion
            vision_obs_enc_flat = vision_obs_enc.view(batch_size, -1)
            depth_obs_enc_flat = depth_obs_enc.view(batch_size, -1)
            vision_goal_enc_flat = vision_goal_enc.view(batch_size, -1)
            depth_goal_enc_flat = depth_goal_enc.view(batch_size, -1)
            
            # Fuse observation features
            if self.use_cross_attention:
                # Add a sequence dimension for attention
                obs_enc = self.vision_depth_fusion(
                    vision_obs_enc_flat.unsqueeze(1), 
                    depth_obs_enc_flat.unsqueeze(1)
                ).squeeze(1)
                goal_enc = self.vision_depth_fusion(
                    vision_goal_enc_flat.unsqueeze(1), 
                    depth_goal_enc_flat.unsqueeze(1)
                ).squeeze(1)
            else:
                obs_enc = self.vision_depth_fusion(torch.cat([vision_obs_enc_flat, depth_obs_enc_flat], dim=-1))
                goal_enc = self.vision_depth_fusion(torch.cat([vision_goal_enc_flat, depth_goal_enc_flat], dim=-1))

            # Store features from the current timestep for contrastive loss
            last_vision_features = vision_obs_enc.view(batch_size, self.context_size, -1)[:, -1, :]
            last_depth_features = depth_obs_enc.view(batch_size, self.context_size, -1)[:, -1, :]

        else: # No depth data, use vision features only
            obs_enc = vision_obs_enc.view(batch_size, -1)
            goal_enc = vision_goal_enc.view(batch_size, -1)
            last_vision_features, last_depth_features = None, None

        # 3. Reshape and pass to transformer decoder
        obs_enc = obs_enc.reshape((batch_size, self.context_size, self.obs_encoding_size))
        goal_enc = goal_enc.unsqueeze(1)
        tokens = torch.cat((obs_enc, goal_enc), dim=1)
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
        
        # Split the context stack into individual images
        obs_list = torch.split(obs_stack, num_channels, dim=1)
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
        
        # Reshape back to (B, context_size, D)
        obs_enc = obs_enc.reshape((batch_size, self.context_size, -1))
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