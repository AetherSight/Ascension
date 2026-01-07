import os
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from lib import SupConClothingDataset, EmbeddingModel, SupConLoss, ClothingTransform


def train(
    data_root: str, 
    batch_size: int = 16, 
    target_batch: int = 128, 
    epochs: int = 50, 
    warmup_epochs: int = 5, 
    lr: float = 3e-4, 
    save_dir: str = "checkpoints",
    resume_path: str | None = None,   # ⭐ 新增
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
        "temperature": 0.1
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
    model = EmbeddingModel(config["model_name"]).to(device)
    criterion = SupConLoss(temperature=config["temperature"])
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
                embeddings = model(views)
                embeddings = F.normalize(embeddings, dim=1)
                loss = criterion(
                    embeddings.view(bsz, 2, -1),
                    labels
                ) / accumulation_steps

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

    print("✅ SupCon 训练完成")


if __name__ == "__main__":
    train(
        data_root="S:\\FFXIV_train_dataset",
        batch_size=16,
        target_batch=128,
        epochs=70,
        warmup_epochs=5,
        lr=3e-4,
        save_dir="checkpoints",
        resume_path="checkpoints/epoch_10_supcon.pth"
    )
