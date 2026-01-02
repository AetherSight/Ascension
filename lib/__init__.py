"""
Ascension 库模块
包含数据集、模型、损失函数和数据变换等核心组件
"""

from .dataset import (
    GalleryDataset,
    ClothingFolderDataset,
    SupConClothingDataset,
    ClothingDataset,
    create_dataloader,
    imread_unicode,
)

from .model import EmbeddingModel
from .loss import SupConLoss
from .transforms import (
    ColorAgnosticTransform,
    ClothingTransform,
    preview_augmentations
)

__all__ = [
    # Dataset
    'GalleryDataset',
    'ClothingFolderDataset',
    'SupConClothingDataset',
    'ClothingDataset',
    'create_dataloader',
    'imread_unicode',
    # Model
    'EmbeddingModel',
    # Loss
    'SupConLoss',
    # Transforms
    'ColorAgnosticTransform',
    'ClothingTransform',
    'preview_augmentations',
]

