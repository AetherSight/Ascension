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

        hsv[..., 0] = (hsv[..., 0] + random.uniform(-30, 30)) % 180
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
        if self.train:
            p = random.random()

            if p < 0.7:
                img = self.random_grayscale(img)
            elif p < 0.9:
                img = self.strong_hsv_jitter(img)
            else:
                img = self.channel_corrupt(img)

        img = cv2.resize(img, (self.img_size, self.img_size))
        img = self.to_tensor(img)
        img = self.normalize(img)

        return img


class StripeDropout(A.DualTransform):
    def __init__(
        self,
        stripe_width_range=(70, 100),
        position='random',
        fill_value=255,
        always_apply=False,
        p=1.0
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.stripe_width_range = stripe_width_range
        self.position = position
        self.fill_value = fill_value

    def apply(self, img, **params):
        h, w = img.shape[:2]
        img = img.copy()
        
        stripe_width = random.randint(self.stripe_width_range[0], self.stripe_width_range[1])
        
        if self.position == 'random':
            rand_val = random.random()
            if rand_val < 0.25:
                position_type = 'top'
            elif rand_val < 0.5:
                position_type = 'bottom'
            elif rand_val < 0.75:
                position_type = 'left'
            else:
                position_type = 'right'
        else:
            position_type = self.position
        
        center_range = 0.8
        margin = (1 - center_range) / 2
        
        if position_type == 'top':
            center_start = int(h * margin)
            center_end = int(h * (margin + center_range * 0.5))
            max_y = max(center_start, center_end - stripe_width)
            y_start = random.randint(center_start, max_y) if max_y >= center_start else center_start
            y_end = min(y_start + stripe_width, h)
            img[y_start:y_end, :] = self.fill_value
        elif position_type == 'bottom':
            center_start = int(h * (margin + center_range * 0.5))
            center_end = int(h * (1 - margin))
            max_y = max(center_start, center_end - stripe_width)
            y_start = random.randint(center_start, max_y) if max_y >= center_start else center_start
            y_end = min(y_start + stripe_width, h)
            img[y_start:y_end, :] = self.fill_value
        elif position_type == 'left':
            center_start = int(w * margin)
            center_end = int(w * (margin + center_range * 0.5))
            max_x = max(center_start, center_end - stripe_width)
            x_start = random.randint(center_start, max_x) if max_x >= center_start else center_start
            x_end = min(x_start + stripe_width, w)
            img[:, x_start:x_end] = self.fill_value
        else:
            center_start = int(w * (margin + center_range * 0.5))
            center_end = int(w * (1 - margin))
            max_x = max(center_start, center_end - stripe_width)
            x_start = random.randint(center_start, max_x) if max_x >= center_start else center_start
            x_end = min(x_start + stripe_width, w)
            img[:, x_start:x_end] = self.fill_value
        
        return img

    def apply_to_mask(self, mask, **params):
        return self.apply(mask, **params)

    def get_transform_init_args_names(self):
        return ('stripe_width_range', 'position', 'fill_value')


class LargeWhiteHole(A.DualTransform):
    """
    模拟抠图导致的白色孔洞（大的不规则白色区域）
    """
    def __init__(
        self,
        min_hole_size=(100, 100),
        max_hole_size=(200, 200),
        num_holes=(1, 3),
        fill_value=255,
        always_apply=False,
        p=1.0
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.min_hole_size = min_hole_size
        self.max_hole_size = max_hole_size
        self.num_holes = num_holes
        self.fill_value = fill_value

    def apply(self, img, **params):
        h, w = img.shape[:2]
        img = img.copy()
        
        num_holes = random.randint(self.num_holes[0], self.num_holes[1])
        
        for _ in range(num_holes):
            hole_w = random.randint(self.min_hole_size[0], self.max_hole_size[0])
            hole_h = random.randint(self.min_hole_size[1], self.max_hole_size[1])
            
            # 确保孔洞在图像范围内
            x = random.randint(0, max(1, w - hole_w))
            y = random.randint(0, max(1, h - hole_h))
            
            # 创建椭圆形或矩形孔洞（更自然）
            if random.random() < 0.5:
                # 椭圆形孔洞
                center_x = x + hole_w // 2
                center_y = y + hole_h // 2
                axes_x = hole_w // 2
                axes_y = hole_h // 2
                cv2.ellipse(img, (center_x, center_y), (axes_x, axes_y), 0, 0, 360, 
                           (self.fill_value, self.fill_value, self.fill_value), -1)
            else:
                # 矩形孔洞
                img[y:y+hole_h, x:x+hole_w] = self.fill_value
        
        return img

    def apply_to_mask(self, mask, **params):
        return self.apply(mask, **params)

    def get_transform_init_args_names(self):
        return ('min_hole_size', 'max_hole_size', 'num_holes', 'fill_value')


class EdgePadding(A.DualTransform):
    """
    模拟抠图不干净，边缘多了部位（在边缘添加随机内容）
    """
    def __init__(
        self,
        padding_range=(20, 80),
        fill_mode='random',  # 'random', 'white', 'noise', 'mirror'
        always_apply=False,
        p=1.0
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.padding_range = padding_range
        self.fill_mode = fill_mode

    def apply(self, img, **params):
        h, w = img.shape[:2]
        
        # 随机选择要扩展的边（可以多边）
        pad_top = random.randint(self.padding_range[0], self.padding_range[1]) if random.random() < 0.5 else 0
        pad_bottom = random.randint(self.padding_range[0], self.padding_range[1]) if random.random() < 0.5 else 0
        pad_left = random.randint(self.padding_range[0], self.padding_range[1]) if random.random() < 0.5 else 0
        pad_right = random.randint(self.padding_range[0], self.padding_range[1]) if random.random() < 0.5 else 0
        
        if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
            return img
        
        # 根据 fill_mode 填充
        if self.fill_mode == 'white':
            fill_value = 255
        elif self.fill_mode == 'random':
            # 随机颜色（模拟其他部位）
            fill_value = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
        elif self.fill_mode == 'noise':
            # 噪声填充
            fill_value = 'noise'
        elif self.fill_mode == 'mirror':
            # 镜像填充
            fill_value = 'mirror'
        else:
            fill_value = 255
        
        if fill_value == 'noise':
            # 噪声填充
            img = cv2.copyMakeBorder(
                img, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=(128, 128, 128)
            )
            # 添加噪声
            noise = np.random.randint(0, 255, (pad_top + h + pad_bottom, pad_left + w + pad_right, 3), dtype=np.uint8)
            mask = np.zeros((pad_top + h + pad_bottom, pad_left + w + pad_right, 3), dtype=np.uint8)
            mask[pad_top:pad_top+h, pad_left:pad_left+w] = 255
            img = np.where(mask == 255, img, noise)
        elif fill_value == 'mirror':
            # 镜像填充
            img = cv2.copyMakeBorder(
                img, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_REFLECT_101
            )
        else:
            # 常量填充
            img = cv2.copyMakeBorder(
                img, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=fill_value
            )
        
        return img

    def apply_to_mask(self, mask, **params):
        return self.apply(mask, **params)

    def get_transform_init_args_names(self):
        return ('padding_range', 'fill_mode')


class ClothingTransform:
    def __init__(self, train=True, return_tensor=True):
        if train:
            base_transforms = [
                # 边缘扩展（模拟抠图不干净，多了部位）
                # 放在最前面，确保后续裁剪能保留部分边缘效果
                EdgePadding(
                    padding_range=(20, 80),
                    fill_mode='random',  # 随机颜色模拟其他部位
                    p=0.4
                ),
                
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

                A.Affine(
                    rotate=(-20, 20),
                    shear=(-8, 8),
                    scale=(0.9, 1.1),
                    translate_percent=(-0.05, 0.05),
                    interpolation=cv2.INTER_NEAREST,
                    mode=cv2.BORDER_CONSTANT,
                    cval=255,
                    p=0.6
                ),

                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ], p=0.4),

                A.OneOf([
                    A.ToGray(p=1.0),
                    A.ChannelShuffle(p=1.0),
                ], p=0.4),

                A.HueSaturationValue(
                    hue_shift_limit=180,
                    sat_shift_limit=(-80, 20),
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

                A.OneOf([
                    A.CoarseDropout(
                        max_holes=8,
                        min_holes=3,
                        max_height=80,
                        max_width=80,
                        fill_value=255,
                        p=1.0
                    ),
                    A.GridDropout(
                        ratio=0.4,
                        unit_size_min=48,
                        unit_size_max=72,
                        fill_value=255,
                        p=1.0
                    ),
                    StripeDropout(
                        stripe_width_range=(70, 100),
                        position='random',
                        fill_value=255,
                        p=1.0
                    ),
                    LargeWhiteHole(
                        min_hole_size=(100, 100),
                        max_hole_size=(200, 200),
                        num_holes=(1, 3),
                        fill_value=255,
                        p=1.0
                    ),
                ], p=0.7),  # 提高概率，因为这是核心问题

                A.OneOf([
                    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.7, 1.0), p=1.0),
                    A.Emboss(alpha=(0.15, 0.4), strength=(0.4, 0.8), p=1.0),
                    A.CLAHE(clip_limit=(2.0, 4.0), tile_grid_size=(8, 8), p=1.0),
                ], p=0.5),

                A.RandomBrightnessContrast(
                    brightness_limit=0.25,
                    contrast_limit=0.25,
                    p=0.4
                ),

                A.RandomGamma(
                    gamma_limit=(80, 130),
                    p=0.3
                ),

                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 30.0), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1.0),
                ], p=0.3),

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
    image = imread_unicode(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = ClothingTransform(train=True, return_tensor=False)
    
    rows, cols = grid_size
    num_images = rows * cols
    
    augmented_images = []
    for i in range(num_images):
        np.random.seed(i)
        augmented = transform(image_rgb)
        augmented_images.append(augmented)
    
    target_h, target_w = 256, 256
    resized_images = []
    for img in augmented_images:
        resized = cv2.resize(img, (target_w, target_h))
        resized_images.append(resized)
    
    grid_h = rows * target_h
    grid_w = cols * target_w
    grid_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    for idx, img in enumerate(resized_images):
        row = idx // cols
        col = idx % cols
        y_start = row * target_h
        y_end = y_start + target_h
        x_start = col * target_w
        x_end = x_start + target_w
        grid_image[y_start:y_end, x_start:x_end] = img
    
    grid_image_bgr = cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR)
    
    if output_path:
        cv2.imwrite(output_path, grid_image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"预览图片已保存到: {output_path}")
    
    if show:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(15, 15))
            plt.imshow(grid_image)
            plt.axis('off')
            plt.title(f'数据增强预览 ({rows}x{cols} 网格)', fontsize=16, pad=20)
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib 未安装，无法显示图片。请安装: pip install matplotlib")
            cv2.imshow('Augmentation Preview', grid_image_bgr)
            print("按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    return grid_image_bgr


if __name__ == "__main__":
    import sys
    image_path = sys.argv[1]
    
    if os.path.exists(image_path):
        print("正在生成数据增强预览...")
        preview_augmentations(
            image_path=image_path,
            grid_size=(5, 5),
            output_path="preview_augmentations.jpg",
            show=True
        )
