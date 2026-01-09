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
            num_classes=0  # 去掉分类头
        )
        
        backbone_dim = self.backbone.num_features
        
        # 全局特征头
        self.global_head = nn.Sequential(
            nn.Linear(backbone_dim, emb_dim),
            nn.BatchNorm1d(emb_dim)
        )
        
        # 局部特征头（与全局共享backbone，但使用独立的head）
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
            # 提取局部特征
            local_emb = self.local_head(feat)
            local_emb = F.normalize(local_emb, dim=1)
            return None, local_emb
        else:
            # 提取全局特征
            global_emb = self.global_head(feat)
            global_emb = F.normalize(global_emb, dim=1)
            return global_emb, None
    
    def extract_patch_features(self, x, patch_size=256, num_patches=4):
        """
        从完整图像中提取局部patch特征
        
        Args:
            x: 完整图像 [B, C, H, W]
            patch_size: patch大小
            num_patches: 提取的patch数量
        
        Returns:
            patch_features: [B*num_patches, emb_dim]
        """
        if not self.use_local_features:
            raise ValueError("use_local_features must be True to extract patch features")
        
        B, C, H, W = x.shape
        device = x.device
        
        # 随机采样patches
        patches = []
        for b in range(B):
            for _ in range(num_patches):
                # 随机位置
                max_top = max(1, H - patch_size + 1)
                max_left = max(1, W - patch_size + 1)
                top = torch.randint(0, max_top, (1,), device=device).item()
                left = torch.randint(0, max_left, (1,), device=device).item()
                
                patch = x[b:b+1, :, top:top+patch_size, left:left+patch_size]
                # 如果patch太小，resize到patch_size
                if patch.shape[2] < patch_size or patch.shape[3] < patch_size:
                    patch = F.interpolate(patch, size=(patch_size, patch_size), mode='bilinear', align_corners=False)
                patches.append(patch.squeeze(0))
        
        patches = torch.stack(patches)  # [B*num_patches, C, patch_size, patch_size]
        
        # 提取局部特征
        _, patch_emb = self.forward(patches, return_local=True)
        return patch_emb


