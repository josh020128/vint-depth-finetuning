"""
FuSe-inspired modules for multimodal fusion
Purpose: Implement contrastive loss and cross-modal attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalContrastiveLoss(nn.Module):
    """FuSe's contrastive loss between vision and depth"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, vision_features, depth_features):
        # Normalize features
        vision_features = F.normalize(vision_features, dim=-1)
        depth_features = F.normalize(depth_features, dim=-1)
        
        # Compute similarity
        logits = torch.matmul(vision_features, depth_features.T) / self.temperature
        
        # Labels are diagonal (positive pairs)
        batch_size = vision_features.shape[0]
        labels = torch.arange(batch_size, device=vision_features.device)
        
        # Symmetric loss
        loss_v2d = F.cross_entropy(logits, labels)
        loss_d2v = F.cross_entropy(logits.T, labels)
        
        return (loss_v2d + loss_d2v) / 2

class CrossModalAttention(nn.Module):
    """Cross-attention for fusing vision and depth"""
    def __init__(self, embed_dim=512, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, vision_feat, depth_feat):
        # Cross-attend vision to depth
        vision_norm = self.norm1(vision_feat)
        depth_norm = self.norm2(depth_feat)
        
        attended, _ = self.attention(
            query=vision_norm,
            key=depth_norm,
            value=depth_norm
        )
        
        return vision_feat + attended  # Residual connection