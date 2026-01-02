import os
import numpy as np
import random
import cv2
from torch.utils.data import DataLoader
from albumentations import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms


def imread_unicode(image_path):
    """
    读取包含中文路径的图片文件（解决 OpenCV 无法读取中文文件名的问题）
    
    Args:
        image_path: 图片路径（可以是包含中文的路径）
    
    Returns:
        numpy array (BGR 格式)，如果读取失败返回 None
    """
    # 使用 numpy.fromfile 读取文件，然后使用 cv2.imdecode 解码
    # 这样可以避免 OpenCV 直接处理中文路径的问题
    img_array = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


class ColorAgnosticTransform:
    def __init__(self, train=True, img_size=224):
        self.train = train
        self.img_size = img_size

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.to_tensor = transforms.ToTensor()

    def random_grayscale(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = np.stack([gray, gray, gray], axis=-1)
        return gray

    def strong_hsv_jitter(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Hue 大幅扰动
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-30, 30)) % 180
        # Saturation & Value 强扰动
        hsv[..., 1] *= random.uniform(0.2, 1.8)
        hsv[..., 2] *= random.uniform(0.5, 1.5)

        hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def channel_corrupt(self, img):
        img = img.copy()
        c = random.randint(0, 2)
        img[..., c] = img[..., random.randint(0, 2)]
        return img

    def __call__(self, img):
        """
        img: RGB numpy array (H, W, 3)
        """
        if self.train:
            p = random.random()

            if p < 0.7:
                img = self.random_grayscale(img)
            elif p < 0.9:
                img = self.strong_hsv_jitter(img)
            else:
                img = self.channel_corrupt(img)

        # resize
        img = cv2.resize(img, (self.img_size, self.img_size))

        # to tensor
        img = self.to_tensor(img)
        img = self.normalize(img)

        return img


class ClothingTransform:
    def __init__(self, train=True, return_tensor=True):
        if train:
            base_transforms = [
                # 局部纹理
                A.OneOf([
                    A.RandomResizedCrop(
                        height=512, width=512, 
                        scale=(0.10, 0.20), 
                        ratio=(0.8, 1.2), 
                        p=1
                    ),
                    A.RandomResizedCrop(
                        height=512, width=512, 
                        scale=(0.15, 0.35), 
                        ratio=(0.75, 1.33), 
                        p=1
                    ),
                ], p=1),

                #A.RandomResizedCrop(
                #    height=511, width=512,
                #    scale=(-1.55, 1.0),     # ⬅️ 更激进，破坏整体轮廓
                #    ratio=(-1.75, 1.25),
                #    p=0.0
                #),
                A.HorizontalFlip(p=0.5),

                A.Affine(
                    rotate=(-20, 20),
                    shear=(-8, 8),
                    scale=(0.9, 1.1),
                    translate_percent=(-0.05, 0.05),
                    p=0.6
                ),

                # 2. 颜色破坏（核心：颜色必须失效）
                A.OneOf([
                    A.ToGray(p=1.0),
                    A.ChannelShuffle(p=1.0),
                ], p=0.5),

                A.HueSaturationValue(
                    hue_shift_limit=180,
                    sat_shift_limit=(-120, 20),  # ⬅️ 更狠，直接打死饱和度
                    val_shift_limit=40,
                    p=0.9
                ),

                A.ColorJitter(
                    brightness=0.6,
                    contrast=0.6,
                    saturation=0.8,
                    hue=0.2,
                    p=0.8
                ),

                A.ChannelDropout(
                    channel_drop_range=(1, 2),   # ⬅️ 允许丢 2 个 channel
                    fill_value=0,
                    p=0.35
                ),

                # 3. 纹理 / 边缘强化（防止颜色破坏后只剩噪声）
                A.OneOf([
                    A.Sharpen(alpha=(0.15, 0.35), lightness=(0.7, 1.0)),
                    A.Emboss(alpha=(0.1, 0.3), strength=(0.3, 0.6)),
                ], p=0.4),

                # 4. 局部遮挡（模拟动作 / 装备叠加）
                A.CoarseDropout(
                    max_holes=9,
                    min_holes=4,
                    max_height=64,
                    max_width=64,
                    fill_value=0,
                    p=0.3
                ),

                # 5. 光照扰动（材质 & 高光随机化）
                A.RandomBrightnessContrast(
                    brightness_limit=0.25,
                    contrast_limit=0.25,
                    p=0.4
                ),

                A.RandomGamma(
                    gamma_limit=(70, 130),
                    p=0.3
                ),

                A.ToGray(p=0.2),

            ]

            base_transforms = [
                # 1️⃣ 语义安全裁剪（绝不裁没装备） 
                A.OneOf([
                    A.RandomResizedCrop(
                        height=512, width=512, 
                        scale=(0.5, 1.0), 
                        ratio=(0.8, 1.2), 
                        p=1
                    ),
                    A.RandomResizedCrop(
                        height=512, width=512, 
                        scale=(0.20, 0.40), 
                        ratio=(0.75, 1.33), 
                        p=1
                    ),
                ], p=0.9),

                A.Resize(512, 512),
                A.HorizontalFlip(p=0.5),

                # 2️⃣ 形变扰动（轻破轮廓，但不换语义）
                A.Affine(
                    rotate=(-20, 20),
                    shear=(-8, 8),
                    scale=(0.9, 1.1),
                    translate_percent=(-0.05, 0.05),
                    p=0.6
                ),

                # 3️⃣ 颜色失效（你这块思路是对的，略微收敛）
                A.OneOf([
                    A.ToGray(p=1.0),
                    A.ChannelShuffle(p=1.0),
                ], p=0.4),

                A.HueSaturationValue(
                    hue_shift_limit=180,
                    sat_shift_limit=(-80, 20),   # ⬅️ 稍微收一点，防止过噪
                    val_shift_limit=40,
                    p=0.8
                ),

                A.ColorJitter(
                    brightness=0.5,
                    contrast=0.5,
                    saturation=0.7,
                    hue=0.15,
                    p=0.8
                ),

                A.ChannelDropout(
                    channel_drop_range=(1, 2),
                    fill_value=0,
                    p=0.3
                ),

                # 4️⃣ 反轮廓核心：局部遮挡（比裁剪重要得多）
                A.OneOf([
                    A.CoarseDropout(
                        max_holes=6,
                        min_holes=3,
                        max_height=80,
                        max_width=80,
                        fill_value=0,
                        p=1.0
                    ),
                    A.GridDropout(
                        ratio=0.4,
                        unit_size_min=32,
                        unit_size_max=64,
                        p=1.0
                    ),
                ], p=0.6),

                # 5️⃣ 纹理 / 边缘强化（防止只剩颜色噪声）
                A.OneOf([
                    A.Sharpen(alpha=(0.15, 0.35), lightness=(0.7, 1.0)),
                    A.Emboss(alpha=(0.1, 0.3), strength=(0.3, 0.6)),
                ], p=0.35),

                # 6️⃣ 光照 & 动态范围扰动
                A.RandomBrightnessContrast(
                    brightness_limit=0.25,
                    contrast_limit=0.25,
                    p=0.4
                ),

                A.RandomGamma(
                    gamma_limit=(80, 130),
                    p=0.3
                ),

                A.ToGray(p=0.15),
            ]
        else:
            base_transforms = [
                A.Resize(512, 512),
            ]

        self.transform = A.Compose(
            base_transforms + (
                [
                    A.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)
                    ),
                    ToTensorV2()
                ] if return_tensor else []
            )
        )

    def __call__(self, image):
        return self.transform(image=image)["image"]


def preview_augmentations(image_path, grid_size=(5, 5), output_path=None, show=True):
    """
    预览数据增强效果，生成网格图片
    
    Args:
        image_path: 输入图片路径
        grid_size: 网格大小，默认为 (5, 5) 即 5x5=25 张图片
        output_path: 保存路径，如果为 None 则不保存
        show: 是否显示图片（使用 matplotlib）
    
    Returns:
        合并后的网格图片（numpy array，BGR 格式）
    """
    # 读取图片（使用支持中文路径的函数）
    image = imread_unicode(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    # 转换为 RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 复用 ClothingTransform，设置 return_tensor=False 用于预览
    transform = ClothingTransform(train=True, return_tensor=False)
    
    rows, cols = grid_size
    num_images = rows * cols
    
    # 生成增强图片
    augmented_images = []
    for i in range(num_images):
        # 设置不同的随机种子以确保每张图片都不同
        np.random.seed(i)
        augmented = transform(image_rgb)
        augmented_images.append(augmented)
    
    # 统一图片尺寸（使用第一张图片的尺寸，或固定尺寸）
    target_h, target_w = 256, 256  # 网格中每张图片的尺寸
    resized_images = []
    for img in augmented_images:
        resized = cv2.resize(img, (target_w, target_h))
        resized_images.append(resized)
    
    # 创建网格
    grid_h = rows * target_h
    grid_w = cols * target_w
    grid_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    # 填充网格
    for idx, img in enumerate(resized_images):
        row = idx // cols
        col = idx % cols
        y_start = row * target_h
        y_end = y_start + target_h
        x_start = col * target_w
        x_end = x_start + target_w
        grid_image[y_start:y_end, x_start:x_end] = img
    
    # 转换为 BGR 格式（用于保存和显示）
    grid_image_bgr = cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR)
    
    # 保存图片
    if output_path:
        cv2.imwrite(output_path, grid_image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"预览图片已保存到: {output_path}")
    
    # 显示图片
    if show:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(15, 15))
            plt.imshow(grid_image)  # matplotlib 使用 RGB
            plt.axis('off')
            plt.title(f'数据增强预览 ({rows}x{cols} 网格)', fontsize=16, pad=20)
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib 未安装，无法显示图片。请安装: pip install matplotlib")
            # 使用 OpenCV 显示（BGR 格式）
            cv2.imshow('Augmentation Preview', grid_image_bgr)
            print("按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    return grid_image_bgr


# 示例使用
if __name__ == "__main__":
    # 预览数据增强效果
    image_path = "images/test2.png"
    
    if os.path.exists(image_path):
        print("正在生成数据增强预览...")
        preview_augmentations(
            image_path=image_path,
            grid_size=(5, 5),
            output_path="preview_augmentations.jpg",
            show=True
        )
        
        # 也可以测试 DataLoader
        print("\n测试 DataLoader...")
        from .dataset import ClothingDataset, create_dataloader
        
        dataset = ClothingDataset(
            image_paths=[image_path] * 100,  # 重复 100 次用于演示
            is_train=True
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0,  # Windows 上建议设为 0
            pin_memory=False
        )
        
        print(f"数据集大小: {len(dataset)}")
        print(f"批次数量: {len(dataloader)}")
        
        # 获取一个批次的数据
        for batch_idx, (images, labels) in enumerate(dataloader):
            print(f"批次 {batch_idx}:")
            print(f"  图片形状: {images.shape}")  # (batch_size, C, H, W)
            print(f"  数据类型: {images.dtype}")
            print(f"  标签数量: {len(labels)}")
            
            if batch_idx >= 2:  # 只显示前 3 个批次
                break
    else:
        print(f"警告: 图片文件 {image_path} 不存在")
        print("请确保 images/test.png 文件存在")
        print("\n使用示例:")
        print("# 预览数据增强:")
        print("preview_augmentations('path/to/image.jpg', grid_size=(5, 5))")
        print("\n# 从文件夹创建 DataLoader:")
        print("from lib import create_dataloader")
        print("dataloader = create_dataloader('path/to/images', batch_size=32)")
        print("\n# 从文件列表创建 DataLoader:")
        print("image_list = ['img1.jpg', 'img2.jpg', ...]")
        print("labels = [0, 1, ...]  # 类别标签")
        print("dataloader = create_dataloader(image_list, labels=labels, batch_size=32)")