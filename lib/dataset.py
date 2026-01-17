import os
import random
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def imread_unicode(image_path: str):
    """Read image from a Unicode path using OpenCV."""
    img_array = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


class GalleryDataset(Dataset):
    """Dataset used to build gallery embeddings."""

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


class MixedSupConClothingDataset(Dataset):
    """
    SupCon dataset that samples from both renderings and real photos.
    - Render images use heavy augmentation.
    - Real images use light augmentation.
    - Sampling probability from the real set is based on real image count.
    """

    def __init__(
        self,
        render_root=r"S:\FFXIV_train_dataset",
        real_root=r"S:\FFXIV_train_dataset2",
        render_transform=None,
        real_transform=None,
        min_real_images=0,
        max_classes=None,
    ):
        self.render_root = Path(render_root)
        self.real_root = Path(real_root)
        self.min_real_images = min_real_images

        if render_transform is None:
            from .transforms import ClothingTransform

            render_transform = ClothingTransform(train=True)
        if real_transform is None:
            from .transforms import RealClothingTransform

            real_transform = RealClothingTransform(train=True)

        self.render_transform = render_transform
        self.real_transform = real_transform

        self.allowed_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

        self.render_map = self._scan_root(self.render_root)
        self.real_map = {
            cls: imgs for cls, imgs in self._scan_root(self.real_root).items()
            if len(imgs) >= self.min_real_images
        }

        self.class_names = sorted(set(self.render_map.keys()) | set(self.real_map.keys()))
        if max_classes is not None:
            self.class_names = self.class_names[:max_classes]
            # trim maps to selected classes
            self.render_map = {cls: imgs for cls, imgs in self.render_map.items() if cls in self.class_names}
            self.real_map = {cls: imgs for cls, imgs in self.real_map.items() if cls in self.class_names}
        if len(self.class_names) == 0:
            raise ValueError("No valid classes found in render or real datasets")
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        self.class_weights = []
        self.class_real_prob = {}
        for cls in self.class_names:
            render_count = len(self.render_map.get(cls, []))
            real_count = len(self.real_map.get(cls, []))
            self.class_real_prob[cls] = self._real_prob(real_count, render_count)
            self.class_weights.append(max(render_count, real_count, 1))

        self.sampled_classes = self._build_sampled_classes()
        self.samples_per_epoch = len(self.sampled_classes)
        render_only = sum(1 for c in self.class_names if c in self.render_map and c not in self.real_map)
        real_only = sum(1 for c in self.class_names if c in self.real_map and c not in self.render_map)
        both = len(self.class_names) - render_only - real_only
        logging.info(
            f"MixedSupConClothingDataset: classes={len(self.class_names)} (render_only={render_only}, "
            f"real_only={real_only}, both={both}), min_real_images={self.min_real_images}, "
            f"max_classes={max_classes}"
        )

    def _scan_root(self, root_dir: Path):
        if not root_dir.exists():
            return {}
        class_map = {}
        for d in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
            imgs = [str(p) for p in d.iterdir() if p.suffix.lower() in self.allowed_exts]
            if len(imgs) > 0:
                class_map[d.name] = imgs
        return class_map

    def _real_prob(self, real_count: int, render_count: int):
        if real_count == 0:
            return 0.0
        if render_count == 0:
            return 1.0
        if real_count < 20:
            return 0.05  # rare real images: show occasionally
        if real_count >= 100:
            return 0.6  # favor real images when abundant
        if real_count >= 50:
            return 1.0 / 3.0  # render:real ~ 2:1
        return 0.25  # render:real ~ 3:1 (20-49)

    def __len__(self):
        return len(self.sampled_classes)

    def _build_sampled_classes(self):
        sampled_classes = []
        for cls, weight in zip(self.class_names, self.class_weights):
            repeat = int(weight)
            if repeat > 0:
                sampled_classes.extend([cls] * repeat)
        random.shuffle(sampled_classes)
        return sampled_classes

    def _pick_image(self, paths):
        return random.choice(paths)

    def __getitem__(self, idx):
        cls = self.sampled_classes[idx]
        render_list = self.render_map.get(cls, [])
        real_list = self.real_map.get(cls, [])

        use_real = False
        if len(real_list) > 0:
            p_real = self.class_real_prob.get(cls, 0.0)
            if random.random() < p_real:
                use_real = True
            elif len(render_list) == 0:
                use_real = True  # fallback if only real exists

        if use_real and len(real_list) > 0:
            img_path = self._pick_image(real_list)
            transform = self.real_transform
        else:
            if len(render_list) == 0:
                raise ValueError(f"Class {cls} has no render or real images available")
            img_path = self._pick_image(render_list)
            transform = self.render_transform

        img = imread_unicode(img_path)
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        v1 = transform(img)
        v2 = transform(img)

        label = self.class_to_idx[cls]
        return torch.stack([v1, v2], dim=0), img, label
