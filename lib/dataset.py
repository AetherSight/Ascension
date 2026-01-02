import os
import numpy as np
import cv2
import random
import torch

from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader


def imread_unicode(image_path):
    img_array = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


class GalleryDataset(Dataset):
    """
    用于构建 gallery embedding 的 Dataset
    """
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        img = imread_unicode(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transform(img)

        return img, self.labels[idx]


class ClothingFolderDataset(Dataset):
    def __init__(self, root_dir, class_names=None, transform=None):
        self.root_dir = Path(root_dir)
        
        all_classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_names = all_classes if class_names is None else class_names
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        
        self.image_paths = []
        self.labels = []
        
        for cls in self.class_names:
            cls_dir = self.root_dir / cls
            if not cls_dir.exists():
                continue
            for img in cls_dir.iterdir():
                if img.suffix.lower() in {".jpg", ".png", ".jpeg", ".webp"}:
                    self.image_paths.append(str(img))
                    self.labels.append(self.class_to_idx[cls])
        
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = imread_unicode(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img, self.labels[idx]


class SupConClothingDataset(ClothingFolderDataset):
    def __init__(self, root, transform):
        super().__init__(root, transform=None)
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = imread_unicode(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # SupCon: two views
        v1 = self.transform(image)
        v2 = self.transform(image)

        return torch.stack([v1, v2], dim=0), label


class ClothingDataset(Dataset):
    """
    服装数据集类，使用 ClothingTransform 进行数据增强
    """
    def __init__(self, image_paths, labels=None, transform=None, is_train=True):
        """
        Args:
            image_paths: 图片路径列表，可以是：
                        - 字符串列表：['path/to/img1.jpg', 'path/to/img2.jpg', ...]
                        - 文件夹路径：'path/to/images'（会自动扫描所有图片）
            labels: 标签列表，如果为 None 则返回图片路径作为标签
            transform: ClothingTransform 实例，如果为 None 则自动创建
            is_train: 是否为训练模式（决定使用哪种 transform）
        """
        # 处理图片路径
        if isinstance(image_paths, str):
            # 如果是文件夹路径，扫描所有图片文件
            if os.path.isdir(image_paths):
                self.image_paths = self._scan_images(image_paths)
            else:
                # 如果是单个文件路径
                self.image_paths = [image_paths] if os.path.exists(image_paths) else []
        elif isinstance(image_paths, list):
            self.image_paths = [p for p in image_paths if os.path.exists(p)]
        else:
            raise ValueError("image_paths 必须是字符串（路径）或字符串列表")
        
        if len(self.image_paths) == 0:
            raise ValueError("未找到任何有效的图片文件")
        
        # 处理标签
        if labels is None:
            # 如果没有提供标签，使用图片路径作为标签（用于无监督学习或测试）
            self.labels = self.image_paths
        else:
            if len(labels) != len(self.image_paths):
                raise ValueError(f"标签数量 ({len(labels)}) 与图片数量 ({len(self.image_paths)}) 不匹配")
            self.labels = labels
        
        # 初始化 transform
        if transform is None:
            from .transforms import ClothingTransform
            self.transform = ClothingTransform(train=is_train)
        else:
            self.transform = transform
    
    def _scan_images(self, folder_path):
        """扫描文件夹中的所有图片文件"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        image_paths = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if os.path.splitext(file.lower())[1] in image_extensions:
                    image_paths.append(os.path.join(root, file))
        return sorted(image_paths)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        返回增强后的图片 tensor 和对应的标签
        
        Returns:
            image: torch.Tensor, shape: (C, H, W)
            label: 标签（可能是路径、类别ID等）
        """
        # 读取图片（使用支持中文路径的函数）
        image_path = self.image_paths[idx]
        image = imread_unicode(image_path)
        
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        # 转换为 RGB（Albumentations 使用 RGB）
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 应用 transform（返回 tensor）
        image_tensor = self.transform(image)
        
        # 确保 tensor 是 float32 类型（ToTensorV2 应该已经转换，但为了安全起见再次确认）
        if image_tensor.dtype != torch.float32:
            image_tensor = image_tensor.float()
        
        # 获取标签
        label = self.labels[idx]
        
        return image_tensor, label
    
    def get_image_path(self, idx):
        """获取指定索引的图片路径"""
        return self.image_paths[idx]


def create_dataloader(image_paths, labels=None, batch_size=32, shuffle=True, 
                     num_workers=4, pin_memory=True, is_train=True, transform=None):
    """
    创建 DataLoader 的便捷函数
    
    Args:
        image_paths: 图片路径（字符串、列表或文件夹路径）
        labels: 标签列表
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 数据加载的进程数
        pin_memory: 是否将数据固定在内存中（GPU 训练时建议为 True）
        is_train: 是否为训练模式
        transform: 自定义 transform，如果为 None 则使用 ClothingTransform
    
    Returns:
        DataLoader 实例
    """
    dataset = ClothingDataset(
        image_paths=image_paths,
        labels=labels,
        transform=transform,
        is_train=is_train
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train  # 训练时丢弃最后一个不完整的 batch
    )
    
    return dataloader

