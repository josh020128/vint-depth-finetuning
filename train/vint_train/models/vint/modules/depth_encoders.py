"""
Depth encoder modules for ViNT
Purpose: Encode depth images into features compatible with ViNT
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet

class DepthEncoder(nn.Module):
    """Base depth encoder using EfficientNet (matching ViNT's approach)"""
    def __init__(self, encoder_name="efficientnet-b0", output_dim=512):
        super().__init__()
        # Use same encoder as ViNT for consistency
        self.encoder = EfficientNet.from_name(encoder_name, in_channels=1)  # 1 channel for depth
        self.num_features = self.encoder._fc.in_features
        
        # Store these for compatibility with ViNT's encoding logic
        self._global_params = self.encoder._global_params
        self._avg_pooling = self.encoder._avg_pooling
        self._dropout = self.encoder._dropout
        
        # Project to match ViNT's dimension
        if self.num_features != output_dim:
            self.projection = nn.Linear(self.num_features, output_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, depth_img):
        # depth_img: [B, 1, H, W]
        features = self.encoder.extract_features(depth_img)
        features = self._avg_pooling(features)
        if self._global_params.include_top:
            features = features.flatten(start_dim=1)
            features = self._dropout(features)
        
        return self.projection(features)
    
    def extract_features(self, depth_img):
        """For compatibility with ViNT's encoding logic"""
        return self.encoder.extract_features(depth_img)