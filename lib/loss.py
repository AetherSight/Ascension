"""
损失函数相关模块
包含 Supervised Contrastive Loss 和 Partial-to-Whole Contrastive Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels=None):
        device = features.device
        if len(features.shape) == 3:
            # 如果是 [B, 2, D]，转为 [2B, D]
            bsz = features.shape[0]
            features = torch.cat([features[:, 0], features[:, 1]], dim=0)
            if labels is not None:
                labels = torch.cat([labels, labels], dim=0)

        # 归一化特征
        features = F.normalize(features, p=2, dim=1)
        
        # 计算相似度矩阵
        batch_size = features.shape[0]
        logits = torch.div(torch.matmul(features, features.T), self.temperature)
        
        # 寻找正样本掩码 (同类为 1)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # 移除自身对比 (Self-mask)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # 计算 log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # 计算平均对数似然
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos.mean()
        
        return loss


class PartialToWholeLoss(nn.Module):
    """
    部分到整体的对比学习损失
    强制局部patch特征与完整图像特征相似
    使用余弦相似度损失，直接最大化相似度
    """
    def __init__(self, temperature=0.1):
        super(PartialToWholeLoss, self).__init__()
        self.temperature = temperature

    def forward(self, global_features, patch_features, labels, num_patches=4):
        """
        Args:
            global_features: 完整图像特征 [B, D]
            patch_features: 局部patch特征 [B*num_patches, D]
            labels: 标签 [B] (当前未使用，保留接口兼容性)
            num_patches: 每个图像的patch数量
        """
        B, D = global_features.shape
        assert patch_features.shape[0] == B * num_patches, \
            "patch_features 的 batch 维度应为 B * num_patches"

        # 归一化
        global_features = F.normalize(global_features, p=2, dim=1)      # [B, D]
        patch_features = F.normalize(patch_features, p=2, dim=1)        # [B*num_patches, D]

        # 将每个 patch 与所有 global 做对比：
        # logits[i, j] = cos(patch_i, global_j) / T
        logits = torch.matmul(patch_features, global_features.t())      # [B*num_patches, B]
        logits = logits / self.temperature

        # 对于第 i 张图的第 k 个 patch（线性索引 i*num_patches + k），
        # 其正样本目标是第 i 个 global_features。
        target_indices = torch.arange(B, device=global_features.device).repeat_interleave(num_patches)

        loss = F.cross_entropy(logits, target_indices)
        return loss

