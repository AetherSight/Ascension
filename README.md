# Ascension

A supervised contrastive learning framework for image recognition model training.

## Overview

Ascension is a deep learning framework designed for training image recognition models using Supervised Contrastive Learning (SupCon). It is particularly suitable for tasks that require learning robust feature representations, such as clothing recognition, equipment identification, and similar image classification/retrieval scenarios.

## Technical Stack

- **Learning Method**: Supervised Contrastive Learning (SupCon Loss)
- **Model Architecture**: EfficientNet series (via `timm` library)
- **Data Augmentation**: Color-agnostic augmentation strategies using Albumentations
- **Training Optimizations**: Gradient accumulation, mixed precision training, learning rate scheduling

## Dataset Organization

Organize your training data in the following directory structure:

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

Each subdirectory represents a class and contains all training images for that class.

## Quick Start

### 1. Install Dependencies

Using Poetry:
```bash
poetry install
```

### 2. Prepare Your Data

Organize your images according to the dataset structure described above.

### 3. Configure Training

Edit `train.py` and modify the `train_supcon()` function call:

```python
if __name__ == "__main__":
    train_supcon(
        data_root="path/to/your/data",
        batch_size=16,
        target_batch=128,  # Effective batch size via gradient accumulation
        epochs=50,
        warmup_epochs=5,
        lr=3e-4,
        save_dir="checkpoints"
    )
```

### 4. Start Training

```bash
python train.py
```

**Key Parameters:**
- `data_root`: Path to your training data root directory (required)
- `batch_size`: Physical batch size (default: 16)
- `target_batch`: Effective batch size via gradient accumulation (default: 128)
- `epochs`: Number of training epochs (default: 50)
- `warmup_epochs`: Learning rate warmup epochs (default: 5)
- `lr`: Initial learning rate (default: 3e-4)
- `save_dir`: Directory to save checkpoints (default: "checkpoints")

The model uses `tf_efficientnetv2_m` as the backbone with a temperature parameter of `0.1` for SupCon loss.

## Requirements

- Python >= 3.13
- CUDA (recommended for GPU training)
- Poetry for dependency management

Main dependencies are managed via `pyproject.toml`:
- PyTorch 2.7.0
- torchvision 0.22.0
- timm (EfficientNet models)
- albumentations (data augmentation)
- opencv-python, numpy, tqdm, matplotlib

## License

Maintained by Aether Sight.
