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


class ProxyAnchorLoss(nn.Module):
    def __init__(self, num_classes, emb_dim=512, margin=0.1, scale=32):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # 每个类一个 proxy（可学习）
        self.proxies = nn.Parameter(
            torch.randn(num_classes, emb_dim)
        )
        nn.init.kaiming_normal_(self.proxies)

    def forward(self, embeddings, labels):
        """
        embeddings: [B, D]  (已 normalize)
        labels:     [B]
        """
        device = embeddings.device
        proxies = F.normalize(self.proxies, dim=1).to(device)

        # cosine similarity: [B, C]
        sim = embeddings @ proxies.t()

        # one-hot labels
        labels_onehot = torch.zeros_like(sim)
        labels_onehot.scatter_(1, labels.view(-1, 1), 1)

        pos_mask = labels_onehot
        neg_mask = 1 - labels_onehot

        pos_exp = torch.exp(-self.scale * (sim - self.margin)) * pos_mask
        neg_exp = torch.exp(self.scale * (sim + self.margin)) * neg_mask

        pos_term = torch.log(1 + pos_exp.sum(dim=0)).mean()
        neg_term = torch.log(1 + neg_exp.sum(dim=0)).mean()

        loss = pos_term + neg_term
        return loss

