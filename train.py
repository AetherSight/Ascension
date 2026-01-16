import os
import numpy as np
import cv2
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# 假设这些库都在 lib 中
from lib import SupConClothingDataset, EmbeddingModel, SupConLoss, PartialToWholeLoss, ClothingTransform, PatchTransform
from evaluate import evaluate_real_world_images


def load_checkpoint(resume_path, model, optimizer, scheduler, warmup_epochs):
    """加载训练断点"""
    print(f"Loading checkpoint: {resume_path}")
    ckpt = torch.load(resume_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    
    start_epoch = ckpt["epoch"] + 1
    best_loss = ckpt.get("loss", float("inf"))
    
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    else:
        if start_epoch > warmup_epochs:
            steps = start_epoch - 1 - warmup_epochs
            for _ in range(max(0, steps)):
                scheduler.step()
    
    print(f"🔁 Resumed from epoch {ckpt['epoch']} -> Start {start_epoch}")
    return start_epoch, best_loss


def extract_and_process_patches(original_images, patch_transform, model, device, patch_size, num_patches):
    """从原图提取patch并应用PatchTransform，返回patch特征 [B*num_patches, D]"""
    # 处理DataLoader可能转换的tensor格式
    if isinstance(original_images, torch.Tensor):
        original_images_np = original_images.cpu().numpy()
        # DataLoader可能将 [B, H, W, 3] 转换为 [B, 3, H, W]，需要转换回来
        if original_images_np.shape[1] == 3 and len(original_images_np.shape) == 4:
            # [B, C, H, W] -> [B, H, W, C]
            original_images_np = original_images_np.transpose(0, 2, 3, 1)
    else:
        original_images_np = original_images
    
    B = original_images_np.shape[0]
    H, W = original_images_np.shape[1], original_images_np.shape[2]
    
    # 确保是uint8格式
    if original_images_np.dtype != np.uint8:
        if original_images_np.max() <= 1.0:
            original_images_np = (original_images_np * 255).astype(np.uint8)
        else:
            original_images_np = original_images_np.astype(np.uint8)
    
    # 中心偏移采样
    center_h = H / 2.0
    center_w = W / 2.0
    
    offset_h_ratios = np.random.uniform(-0.20, 0.20, size=(B, num_patches))
    offset_w_ratios = np.random.uniform(-0.1, 0.1, size=(B, num_patches))
    
    offset_h = offset_h_ratios * H
    offset_w = offset_w_ratios * W
    
    patch_center_h = center_h + offset_h
    patch_center_w = center_w + offset_w
    
    top_positions = (patch_center_h - patch_size / 2.0).astype(np.int32)
    left_positions = (patch_center_w - patch_size / 2.0).astype(np.int32)
    
    top_positions = np.clip(top_positions, 0, H - patch_size)
    left_positions = np.clip(left_positions, 0, W - patch_size)
    
    # 提取patch并应用transform
    patches_list = []
    
    for b_idx in range(B):
        for p_idx in range(num_patches):
            top = top_positions[b_idx, p_idx]
            left = left_positions[b_idx, p_idx]
            
            patch = original_images_np[b_idx, top:top+patch_size, left:left+patch_size]
            
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
            
            patch_tensor = patch_transform(patch)
            patches_list.append(patch_tensor)
    
    patches = torch.stack(patches_list).to(device)
    _, patch_emb = model(patches, return_local=True)
    
    del patches, original_images_np
    return patch_emb


def save_checkpoint(checkpoint, save_dir, epoch, best_loss):
    """保存checkpoint"""
    if checkpoint["loss"] < best_loss:
        best_loss = checkpoint["loss"]
        torch.save(checkpoint, os.path.join(save_dir, "best_supcon.pth"))
    
    torch.save(checkpoint, os.path.join(save_dir, f"epoch_{epoch}_supcon.pth"))
    return best_loss


def run_evaluation(model, eval_gallery_root, eval_image_paths, device, eval_top_k, save_dir, epoch):
    """运行评估"""
    print(f"\n{'='*40}\n📊 Evaluating Epoch {epoch}\n{'='*40}")
    
    torch.cuda.empty_cache()
    
    try:
        cache_path = os.path.join(save_dir, f"epoch_{epoch}_gallery_cache.pth")
        
        with torch.no_grad():
            results = evaluate_real_world_images(
                model=model,
                gallery_root=eval_gallery_root,
                image_paths=eval_image_paths,
                device=device,
                top_k=eval_top_k,
                cache_path=cache_path,
                patch_weight=0.3,
                patch_only=False
            )
        
        for img_path, top_results in results:
            print(f"\n[Query] {os.path.basename(img_path)}")
            for k, (label, score) in enumerate(top_results, 1):
                print(f"  Top-{k}: {label} ({score:.4f})")
        print("\n")
        
    except Exception as e:
        print(f"⚠️ Eval Failed: {e}")
    
    torch.cuda.empty_cache()

def train(
    data_root: str, 
    batch_size: int = 16, 
    target_batch: int = 128, 
    epochs: int = 50, 
    warmup_epochs: int = 5, 
    lr: float = 3e-4, 
    save_dir: str = "checkpoints",
    resume_path: str | None = None,
    eval_gallery_root: str | None = None,
    eval_image_paths: list[str] | None = None,
    eval_top_k: int = 5,
):
    # --- 配置区域 ---
    config = {
        "model_name": "tf_efficientnetv2_m",
        "temperature": 0.1,
        "use_partial_loss": True,
        "partial_loss_weight": 0.5,
        "partial_loss_interval": 4,  # 每4步计算一次局部Loss以节省显存
        "num_patches": 3,
        "patch_size": 224  # 稍微调小一点 (原240) 以防OOM
    }

    accumulation_steps = max(1, target_batch // batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    # --- 数据准备 ---
    train_ds = SupConClothingDataset(
        data_root, 
        transform=ClothingTransform(train=True)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    model = EmbeddingModel(config["model_name"], use_local_features=True).to(device)
    criterion = SupConLoss(temperature=config["temperature"])
    partial_criterion = PartialToWholeLoss(temperature=config["temperature"])
    patch_transform = PatchTransform(return_tensor=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs,
        eta_min=1e-6
    )
    
    scaler = GradScaler("cuda")

    start_epoch = 1
    best_loss = float("inf")

    # --- 断点恢复 ---
    if resume_path is not None:
        start_epoch, best_loss = load_checkpoint(resume_path, model, optimizer, main_scheduler, warmup_epochs)

    print(f"🚀 Phys Batch: {batch_size} | Acc Steps: {accumulation_steps} | Eff Batch: {batch_size * accumulation_steps}")

    for epoch in range(start_epoch, epochs + 1):
        
        if epoch <= warmup_epochs:
            curr_lr = lr * (epoch / warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = curr_lr

        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        last_partial_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for i, (views, original_images, labels) in enumerate(pbar):
            bsz = views.size(0)
            # views: [B, 2, C, H, W] -> Flatten -> [2B, C, H, W]
            images = views.view(-1, *views.shape[2:]).to(device)
            labels = labels.to(device)

            with autocast("cuda"):
                global_emb, _ = model(images, return_local=False)
                global_emb = F.normalize(global_emb, dim=1)
                
                global_emb_stacked = global_emb.view(bsz, 2, -1)
                
                global_loss = criterion(global_emb_stacked, labels)
                
                partial_loss = torch.tensor(0.0, device=device)
                calc_partial = config["use_partial_loss"] and (i % config["partial_loss_interval"] == 0)

                if calc_partial:
                    view2_global = global_emb_stacked[:, 1]
                    
                    patch_emb = extract_and_process_patches(
                        original_images,
                        patch_transform,
                        model,
                        device,
                        config["patch_size"],
                        config["num_patches"]
                    )
                    
                    partial_loss = partial_criterion(
                        view2_global,
                        patch_emb,
                        labels,
                        num_patches=config["num_patches"]
                    )
                    last_partial_loss = partial_loss.item()
                    
                    del patch_emb
                    torch.cuda.empty_cache()
                
                loss = (global_loss + config["partial_loss_weight"] * partial_loss) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            current_loss_val = loss.item() * accumulation_steps
            total_loss += current_loss_val
            
            pbar_dict = {
                "loss": f"{current_loss_val:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                "pl": f"{last_partial_loss:.2f}"
            }
            
            pbar.set_postfix(pbar_dict)

        # Scheduler Step
        if epoch > warmup_epochs:
            main_scheduler.step()

        avg_loss = total_loss / len(train_loader)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": main_scheduler.state_dict(),
            "class_names": train_ds.class_names,
            "epoch": epoch,
            "loss": avg_loss
        }

        best_loss = save_checkpoint(checkpoint, save_dir, epoch, best_loss)
        
        if eval_gallery_root and eval_image_paths:
            run_evaluation(model, eval_gallery_root, eval_image_paths, device, eval_top_k, save_dir, epoch)

    print("✅ SupCon Training Finished")


if __name__ == "__main__":
    train(
        data_root="S:\\FFXIV_train_dataset",
        batch_size=12,
        target_batch=96,
        epochs=70,
        warmup_epochs=5,
        lr=3e-4,
        save_dir="checkpoints_0.0.3",
        resume_path="checkpoints_0.0.3/epoch_30_supcon.pth",
    )
