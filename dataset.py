import numpy as np
import cv2
import torch

from pathlib import Path
from torch.utils.data import Dataset


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

