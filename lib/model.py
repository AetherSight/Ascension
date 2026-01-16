"""Model modules including the EfficientNet-based embedding model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class EmbeddingModel(nn.Module):
    """
    EfficientNet-based embedding model supporting global and optional local (patch) features.
    """

    def __init__(self, model_name="tf_efficientnetv2_m", emb_dim=512, use_local_features=True):
        """
        Args:
            model_name: timm model name
            emb_dim: embedding dimension
            use_local_features: enable local features for patch-to-whole contrast
        """
        super().__init__()
        self.use_local_features = use_local_features
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
        )

        backbone_dim = self.backbone.num_features

        self.global_head = nn.Sequential(
            nn.Linear(backbone_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
        )

        if use_local_features:
            self.local_head = nn.Sequential(
                nn.Linear(backbone_dim, emb_dim),
                nn.BatchNorm1d(emb_dim),
            )

    def forward(self, x, return_local=False):
        """
        Args:
            x: input tensor [B, C, H, W] or [B*N, C, H, W] for patches
            return_local: whether to return local features (used during patch-to-whole training)

        Returns:
            If return_local=False: normalized global embeddings [B, emb_dim]
            If return_local=True: (global_emb, local_emb)
                - global_emb: [B, emb_dim]
                - local_emb: [B*N, emb_dim] or None
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
        """Sample patches around the center and extract their embeddings."""
        if not self.use_local_features:
            raise ValueError("use_local_features must be True to extract patch features")

        B, _, H, W = x.shape
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

            patch = x[b_idx:b_idx + 1, :, top:top + patch_size, left:left + patch_size]

            if patch.shape[2] < patch_size or patch.shape[3] < patch_size:
                patch = F.interpolate(patch, size=(patch_size, patch_size), mode="bilinear", align_corners=False)

            patches_list.append(patch.squeeze(0))

        patches = torch.stack(patches_list)
        _, patch_emb = self.forward(patches, return_local=True)
        return patch_emb
