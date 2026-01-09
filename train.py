import os
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from lib import SupConClothingDataset, EmbeddingModel, SupConLoss, PartialToWholeLoss, ClothingTransform
from evaluate import evaluate_real_world_images


def train(
    data_root: str, 
    batch_size: int = 16, 
    target_batch: int = 128, 
    epochs: int = 50, 
    warmup_epochs: int = 5, 
    lr: float = 3e-4, 
    save_dir: str = "checkpoints",
    resume_path: str | None = None,
    # 评估相关参数
    eval_gallery_root: str | None = None,  # gallery 路径
    eval_image_paths: list[str] | None = None,  # 测试图片路径列表
    eval_top_k: int = 5,  # 评估时输出的 top-k
):
    config = {
        "data_root": data_root,
        "batch_size": batch_size,
        "target_batch": target_batch,
        "epochs": epochs,
        "warmup_epochs": warmup_epochs,
        "lr": lr,
        "save_dir": save_dir,
        "model_name": "tf_efficientnetv2_m",
        "temperature": 0.1,
        "use_partial_loss": True,
        "partial_loss_weight": 0.5,
        "num_patches": 4,
        "patch_size": 256
    }

    accumulation_steps = max(1, config["target_batch"] // config["batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config["save_dir"], exist_ok=True)

    # Dataset & Loader
    train_ds = SupConClothingDataset(
        config["data_root"], 
        transform=ClothingTransform(train=True)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # Model
    model = EmbeddingModel(config["model_name"], use_local_features=True).to(device)
    criterion = SupConLoss(temperature=config["temperature"])
    partial_criterion = PartialToWholeLoss(temperature=config["temperature"])
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"] - config["warmup_epochs"]
    )
    scaler = GradScaler("cuda")

    start_epoch = 1
    best_loss = float("inf")

    # =========================
    # Resume logic
    # =========================
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("loss", best_loss)
        
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            main_scheduler.load_state_dict(ckpt["scheduler"])
        else:
            # Old checkpoint without scheduler state, manually restore
            # Scheduler starts after warmup, so steps = epoch - warmup_epochs
            if start_epoch > config["warmup_epochs"]:
                scheduler_steps = ckpt["epoch"] - config["warmup_epochs"]
                for _ in range(scheduler_steps):
                    main_scheduler.step()
                print(f"⚠️  Old checkpoint detected, manually restored scheduler to step {scheduler_steps}")

        print(f"🔁 Resume SupCon from epoch {ckpt['epoch']} → {start_epoch}")
        print(f"📊 Current LR: {optimizer.param_groups[0]['lr']:.2e}")

    print(
        f"🚀 Physical Batch: {config['batch_size']} | "
        f"Acc steps: {accumulation_steps} | "
        f"Effective Batch: {config['batch_size'] * accumulation_steps}"
    )

    # =========================
    # Training loop
    # =========================
    for epoch in range(start_epoch, config["epochs"] + 1):

        # Warmup（只在真实前几轮）
        if epoch <= config["warmup_epochs"]:
            curr_lr = config["lr"] * (epoch / config["warmup_epochs"])
            for pg in optimizer.param_groups:
                pg["lr"] = curr_lr

        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}")

        for i, (views, labels) in enumerate(pbar):
            bsz = views.size(0)
            views = views.view(-1, *views.shape[2:]).to(device)
            labels = labels.to(device)

            with autocast("cuda"):
                # 提取全局特征
                global_emb, _ = model(views, return_local=False)
                global_emb = F.normalize(global_emb, dim=1)
                
                # 全局对比损失
                global_loss = criterion(
                    global_emb.view(bsz, 2, -1),
                    labels
                )
                
                # 部分到整体对比损失
                partial_loss = 0.0
                if config.get("use_partial_loss", True) and (i % 2 == 0):  # 每2步计算一次，减少计算量
                    # 从原始完整图像提取patch特征
                    # views是增强后的，我们需要从原始batch中提取
                    # 为了简化，我们从views中提取（虽然已经增强，但仍然是局部到整体的关系）
                    patch_emb = model.extract_patch_features(
                        views.view(bsz, 2, *views.shape[1:])[:, 0],  # 使用第一个view
                        patch_size=config.get("patch_size", 256),
                        num_patches=config.get("num_patches", 4)
                    )
                    
                    # 计算部分到整体损失（使用第一个view的全局特征）
                    global_for_partial = global_emb.view(bsz, 2, -1)[:, 0]  # [B, D]
                    labels_for_partial = labels
                    partial_loss = partial_criterion(
                        global_for_partial,
                        patch_emb,
                        labels_for_partial,
                        num_patches=config.get("num_patches", 4)
                    )
                
                # 总损失
                loss_weight = config.get("partial_loss_weight", 0.5)
                loss = (global_loss + loss_weight * partial_loss) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * accumulation_steps
            pbar.set_postfix({
                "loss": f"{loss.item() * accumulation_steps:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        if epoch > config["warmup_epochs"]:
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

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                checkpoint,
                os.path.join(config["save_dir"], "best_supcon.pth")
            )

        if epoch % 5 == 0:
            torch.save(
                checkpoint,
                os.path.join(
                    config["save_dir"], f"epoch_{epoch}_supcon.pth"
                )
            )
            
            # 每5轮评估真实图片效果
            if eval_gallery_root and eval_image_paths:
                print(f"\n{'='*60}")
                print(f"📊 Epoch {epoch} - 评估真实图片效果")
                print(f"{'='*60}")
                
                try:
                    cache_path = os.path.join(config["save_dir"], f"epoch_{epoch}_gallery_cache.pth")
                    results = evaluate_real_world_images(
                        model=model,
                        gallery_root=eval_gallery_root,
                        image_paths=eval_image_paths,
                        device=device,
                        top_k=eval_top_k,
                        cache_path=cache_path
                    )
                    
                    for img_path, top_results in results:
                        print(f"\n[Query] {os.path.basename(img_path)}")
                        for i, (label, score) in enumerate(top_results, 1):
                            print(f"  Top-{i}: {label} (cos={score:.4f})")
                    
                    print(f"{'='*60}\n")
                except Exception as e:
                    print(f"⚠️  评估失败: {e}\n")

    print("✅ SupCon 训练完成")


if __name__ == "__main__":
    # 测试图片路径（可根据需要修改）
    test_images = [
        r"S:\FFXIV_train_test\a.JPG",
        r"S:\FFXIV_train_test\b.JPG",
        r"S:\FFXIV_train_test\c.JPG",
        r"S:\FFXIV_train_test\d.JPG",
        r"S:\FFXIV_train_test\e.JPG",
        r"S:\FFXIV_train_test\1.JPG",
        r"S:\FFXIV_train_test\1_back.JPG",
        r"S:\FFXIV_train_test\1_front.JPG",
        r"S:\FFXIV_train_test\1_front.png",
        r"S:\FFXIV_train_test\1_side.JPG",
        r"S:\FFXIV_train_test\1_part.JPG",
        r"S:\FFXIV_train_test\2.JPG",
        r"S:\FFXIV_train_test\4.JPG",
        r"S:\FFXIV_train_test\4_2.JPG",
        r"S:\FFXIV_train_test\5.JPG",
        r"S:\FFXIV_train_test\6.JPG",
        r"S:\FFXIV_train_test\unknown_1.JPG",
        r"S:\FFXIV_train_test\鬼师.png",
        r"S:\FFXIV_train_test\玉韦亚瓦塔强袭短衣.png",
        r"S:\FFXIV_train_test\download.png",
    ]
    
    train(
        data_root="S:\\FFXIV_train_dataset",
        batch_size=16,
        target_batch=128,
        epochs=70,
        warmup_epochs=5,
        lr=3e-4,
        save_dir="checkpoints",
        # 评估配置（设置为 None 可禁用评估）
        eval_gallery_root=r"S:\FFXIV_train_dataset",
        eval_image_paths=test_images,
        eval_top_k=5,
    )
