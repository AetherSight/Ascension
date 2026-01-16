"""Loss modules: Supervised Contrastive Loss and Partial-to-Whole Contrastive Loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised contrastive loss."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels=None):
        device = features.device
        if len(features.shape) == 3:
            # If [B, 2, D], flatten to [2B, D]
            bsz = features.shape[0]
            features = torch.cat([features[:, 0], features[:, 1]], dim=0)
            if labels is not None:
                labels = torch.cat([labels, labels], dim=0)

        # Normalize features
        features = F.normalize(features, p=2, dim=1)

        # Similarity matrix
        batch_size = features.shape[0]
        logits = torch.div(torch.matmul(features, features.T), self.temperature)

        # Positive mask (same class)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Remove self-comparisons
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # Log-probability
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Mean log-likelihood over positives
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos.mean()
        return loss


class PartialToWholeLoss(nn.Module):
    """Contrastive loss that aligns patch features to full-image features."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, global_features, patch_features, labels, num_patches: int = 4):
        """
        Args:
            global_features: full-image features [B, D]
            patch_features: local patch features [B*num_patches, D]
            labels: labels [B] (kept for API compatibility; unused)
            num_patches: number of patches per image
        """
        B, _ = global_features.shape
        assert patch_features.shape[0] == B * num_patches, "patch_features batch dim must be B * num_patches"

        global_features = F.normalize(global_features, p=2, dim=1)  # [B, D]
        patch_features = F.normalize(patch_features, p=2, dim=1)  # [B*num_patches, D]

        logits = torch.matmul(patch_features, global_features.t())  # [B*num_patches, B]
        logits = logits / self.temperature

        target_indices = torch.arange(B, device=global_features.device).repeat_interleave(num_patches)

        loss = F.cross_entropy(logits, target_indices)
        return loss
