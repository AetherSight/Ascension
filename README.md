# Ascension

一个基于监督对比学习（Supervised Contrastive Learning）的图像识别模型训练框架。

## 项目简介

Ascension 是一个用于训练图像识别模型的深度学习框架，主要采用 Supervised Contrastive Learning (SupCon) 方法。该项目特别适用于需要学习鲁棒特征表示的图像分类和检索任务，如服装识别、装备识别等场景。

## 主要特性

- 🎯 **监督对比学习**: 使用 SupCon Loss 训练，学习更具区分性的特征表示
- 🚀 **高效模型**: 基于 EfficientNet 系列模型，支持多种预训练 backbone
- 🔄 **强大的数据增强**: 针对颜色无关特征学习的数据增强策略
- 📊 **灵活的检索系统**: 支持 gallery-based 图像检索和 Top-K 匹配
- ⚡ **训练优化**: 支持梯度累积、混合精度训练、学习率调度等优化策略

## 安装

### 环境要求

- Python >= 3.13
- CUDA (推荐，用于 GPU 训练)

### 安装依赖

使用 Poetry（推荐）:
```bash
poetry install
```

或使用 pip:
```bash
pip install -r requirements.txt
```

主要依赖：
- PyTorch >= 2.7.0
- torchvision >= 0.22.0
- timm (用于 EfficientNet 模型)
- albumentations (用于数据增强)
- opencv-python
- numpy
- tqdm

## 使用方法

### 数据准备

将训练数据组织为以下目录结构：
```
data_root/
├── class1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── class2/
│   ├── img1.jpg
│   └── ...
└── ...
```

每个子目录代表一个类别，包含该类别的所有训练图像。

### 训练模型

修改 `train.py` 中的配置参数：

```python
config = {
    "data_root": "path/to/your/data",
    "batch_size": 16,
    "target_batch": 128,  # 通过梯度累积达到的有效 batch size
    "epochs": 50,
    "warmup_epochs": 5,
    "lr": 3e-4,
    "save_dir": "checkpoints_supcon",
    "model_name": "tf_efficientnetv2_m",  # timm 模型名称
    "temperature": 0.1  # SupCon loss 温度参数
}
```

运行训练：
```bash
python train.py
```

### 测试和检索

使用训练好的模型进行图像检索：

```python
from test_model import verify_real_world_image

verify_real_world_image(
    model_path="checkpoints_supcon/best_supcon.pth",
    gallery_root="path/to/gallery/images",
    image_paths=["path/to/query/image.jpg"],
    top_k=5
)
```

### 预览数据增强

查看数据增强效果：

```python
from augment_images import preview_augmentations

preview_augmentations(
    image_path="path/to/image.jpg",
    grid_size=(5, 5),
    output_path="preview_augmentations.jpg",
    show=True
)
```

## 核心组件

### EmbeddingModel

基于 EfficientNet 的特征提取模型，输出归一化的嵌入向量。

```python
from model import EmbeddingModel

model = EmbeddingModel(model_name="tf_efficientnetv2_m", emb_dim=512)
```

### SupConLoss

监督对比损失函数，通过拉近同类样本、推远异类样本来学习特征表示。

```python
from loss import SupConLoss

criterion = SupConLoss(temperature=0.1)
```

### A2ClothingTransform

针对颜色无关特征学习的数据增强策略，包括：
- 颜色抖动和通道操作
- 几何变换（旋转、缩放、裁剪）
- 局部遮挡
- 纹理增强

## 配置说明

### 训练参数

- `batch_size`: 物理 batch size
- `target_batch`: 通过梯度累积达到的有效 batch size
- `temperature`: SupCon loss 的温度参数，控制相似度分布的锐度
- `warmup_epochs`: 学习率预热轮数
- `lr`: 初始学习率

### 模型选择

支持所有 timm 库中的模型，推荐使用：
- `tf_efficientnetv2_m` (默认)
- `tf_efficientnetv2_s`
- `tf_efficientnetv2_l`

## 工具脚本

### 清理数据目录

清理图片数量过少的类别目录：

```bash
python scripts/cleanup_low_count_dirs.py --data-root path/to/data --min-images 70 --execute
```

## 许可证

本项目由 Aether Sight 组织维护。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 作者

Aether Sight

