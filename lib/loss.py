"""
损失函数相关模块
包含 Supervised Contrastive Loss
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

