"""
模型相关模块
包含 EmbeddingModel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class EmbeddingModel(nn.Module):
    """
    基于 EfficientNet 的嵌入模型
    支持全局特征和局部patch特征提取，用于局部到整体的推断
    """
    def __init__(self, model_name="tf_efficientnetv2_m", emb_dim=512, use_local_features=True):
        """
        Args:
            model_name: timm 模型名称
            emb_dim: 嵌入维度
            use_local_features: 是否使用局部特征（用于局部到整体推断）
        """
        super().__init__()
        self.use_local_features = use_local_features
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0
        )
        
        backbone_dim = self.backbone.num_features
        
        self.global_head = nn.Sequential(
            nn.Linear(backbone_dim, emb_dim),
            nn.BatchNorm1d(emb_dim)
        )
        
        if use_local_features:
            self.local_head = nn.Sequential(
                nn.Linear(backbone_dim, emb_dim),
                nn.BatchNorm1d(emb_dim)
            )
    
    def forward(self, x, return_local=False):
        """
        Args:
            x: 输入图像张量 [B, C, H, W] 或 [B*N, C, H, W] (N个patches)
            return_local: 是否返回局部特征（用于训练时的局部到整体对比）
        
        Returns:
            如果 return_local=False: 归一化的全局嵌入向量 [B, emb_dim]
            如果 return_local=True: (global_emb, local_emb)
                - global_emb: [B, emb_dim]
                - local_emb: [B*N, emb_dim] 或 None
        """
        feat = self.backbone(x)
        
        if return_local and self.use_local_features:
            local_emb = self.local_head(feat)
            local_emb = F.normalize(local_emb, dim=1)
            return None, local_emb
        else:
            global_emb = self.global_head(feat)
            global_emb = F.normalize(global_emb, dim=1)
            return global_emb, None
    
    def extract_patch_features(self, x, patch_size=256, num_patches=4):
        if not self.use_local_features:
            raise ValueError("use_local_features must be True to extract patch features")
        
        B, C, H, W = x.shape
        device = x.device
        
        center_h = H / 2.0
        center_w = W / 2.0
        
        offset_h_ratios = torch.empty(B, num_patches, device=device).uniform_(-0.20, 0.20)
        offset_w_ratios = torch.empty(B, num_patches, device=device).uniform_(-0.1, 0.1)
        
        offset_h = offset_h_ratios * H
        offset_w = offset_w_ratios * W
        
        patch_center_h = center_h + offset_h
        patch_center_w = center_w + offset_w
        
        top_positions = (patch_center_h - patch_size / 2.0).long()
        left_positions = (patch_center_w - patch_size / 2.0).long()
        
        top_positions = torch.clamp(top_positions, 0, H - patch_size)
        left_positions = torch.clamp(left_positions, 0, W - patch_size)
        
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, num_patches)
        batch_indices = batch_indices.flatten()
        top_positions_flat = top_positions.flatten()
        left_positions_flat = left_positions.flatten()
        
        patches_list = []
        for i in range(B * num_patches):
            b_idx = batch_indices[i]
            top = top_positions_flat[i].item()
            left = left_positions_flat[i].item()
            
            patch = x[b_idx:b_idx+1, :, top:top+patch_size, left:left+patch_size]
            
            if patch.shape[2] < patch_size or patch.shape[3] < patch_size:
                patch = F.interpolate(patch, size=(patch_size, patch_size), mode='bilinear', align_corners=False)
            
            patches_list.append(patch.squeeze(0))
        
        patches = torch.stack(patches_list)
        _, patch_emb = self.forward(patches, return_local=True)
        return patch_emb


