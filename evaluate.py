import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import cv2

from lib import EmbeddingModel, imread_unicode, GalleryDataset, ClothingTransform
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast



'''
# =========================================================
# 构建 / 加载 Gallery Embeddings
# =========================================================
@torch.no_grad()
def build_gallery(
    model,
    gallery_root,
    transform,
    device,
    batch_size=32,
    cache_path=None
):
    # ---------- 直接使用缓存 ----------
    if cache_path and os.path.exists(cache_path):
        print(f"[Gallery] Loading cache from {cache_path}")
        data = torch.load(cache_path, map_location="cpu")
        return data["embs"], data["labels"]

    model.eval()

    image_paths = []
    image_labels = []

    class_names = sorted(
        [d.name for d in os.scandir(gallery_root) if d.is_dir()]
    )

    for cls in class_names:
        cls_dir = os.path.join(gallery_root, cls)
        for img_name in os.listdir(cls_dir):
            if img_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                image_paths.append(os.path.join(cls_dir, img_name))
                image_labels.append(cls)

    print(f"[Gallery] Total images: {len(image_paths)}")

    gallery_embs = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Building gallery"):
        batch_imgs = []
        for p in image_paths[i:i + batch_size]:
            img = imread_unicode(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            batch_imgs.append(transform(img))

        batch = torch.stack(batch_imgs).to(device, non_blocking=True)
        emb = model(batch).cpu()
        gallery_embs.append(emb)

    gallery_embs = torch.cat(gallery_embs, dim=0)
    gallery_embs = F.normalize(gallery_embs, dim=1)

    if cache_path:
        torch.save(
            {"embs": gallery_embs, "labels": image_labels},
            cache_path
        )
        print(f"[Gallery] Cache saved to {cache_path}")

    return gallery_embs, image_labels
'''


@torch.no_grad()
def build_gallery(
    model,
    gallery_root,
    transform,
    device,
    batch_size=128,
    cache_path=None,
    num_workers=8
):
    # --------------------
    # 1. cache
    # --------------------
    if cache_path and os.path.exists(cache_path):
        print(f"[Gallery] Loading cache from {cache_path}")
        data = torch.load(cache_path, map_location="cpu")
        return data["embs"], data["labels"]

    model.eval()

    # --------------------
    # 2. 扫描 gallery
    # --------------------
    image_paths = []
    image_labels = []

    class_names = sorted(
        d.name for d in os.scandir(gallery_root) if d.is_dir()
    )

    for cls in class_names:
        cls_dir = os.path.join(gallery_root, cls)
        for name in os.listdir(cls_dir):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                image_paths.append(os.path.join(cls_dir, name))
                image_labels.append(cls)

    print(f"[Gallery] Total images: {len(image_paths)}")

    # --------------------
    # 3. Dataset / Loader
    # --------------------
    dataset = GalleryDataset(
        image_paths=image_paths,
        labels=image_labels,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    # --------------------
    # 4. 推理
    # --------------------
    gallery_embs = []
    gallery_labels_out = []

    for imgs, labels in tqdm(loader, desc="Building gallery"):
        imgs = imgs.to(device, non_blocking=True)

        with autocast(enabled=(device.type == "cuda")):
            emb = model(imgs)

        emb = F.normalize(emb, dim=1)

        gallery_embs.append(emb.cpu())
        gallery_labels_out.extend(labels)

    gallery_embs = torch.cat(gallery_embs, dim=0)

    # --------------------
    # 5. cache
    # --------------------
    if cache_path:
        torch.save(
            {
                "embs": gallery_embs,
                "labels": gallery_labels_out,
            },
            cache_path
        )
        print(f"[Gallery] Cache saved to {cache_path}")

    return gallery_embs, gallery_labels_out


# =========================================================
# 实拍图 Top-K 检索（使用模型对象）
# =========================================================
@torch.no_grad()
def evaluate_real_world_images(
    model,
    gallery_root,
    image_paths,
    device,
    top_k=5,
    top_n=50,
    cache_path=None
):
    """
    使用模型对象直接评估真实图片（用于训练过程中的评估）
    
    参数：
        model: EmbeddingModel 实例（已加载到 device）
        gallery_root: gallery 图片根目录
        image_paths: list, 待查询图片路径
        device: torch.device
        top_k: int, 最终输出 top-K
        top_n: int, 去重前取的 top-N
        cache_path: str, gallery cache 路径（可选）
    """
    model.eval()
    transform = ClothingTransform(train=False)
    
    # 构建 gallery
    gallery_embs, gallery_labels = build_gallery(
        model,
        gallery_root,
        transform,
        device,
        batch_size=128,
        num_workers=4,  # 训练时减少 workers
        cache_path=cache_path
    )
    
    # 查询并输出结果
    results = []
    for img_path in image_paths:
        if not os.path.exists(img_path):
            continue
            
        img = imread_unicode(img_path)
        if img is None:
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        query = transform(img).unsqueeze(0).to(device)
        query_emb = model(query).cpu()
        query_emb = F.normalize(query_emb, dim=1)
        
        sims = torch.matmul(query_emb, gallery_embs.T)[0]
        vals, idxs = torch.topk(sims, min(top_n, sims.size(0)))
        
        seen = {}
        for v, idx in zip(vals.tolist(), idxs.tolist()):
            label = gallery_labels[idx]
            if label not in seen or v > seen[label]:
                seen[label] = v
        
        if len(seen) < top_k:
            all_idxs = torch.argsort(sims, descending=True)
            for idx in all_idxs.tolist():
                label = gallery_labels[idx]
                if label not in seen:
                    seen[label] = sims[idx].item()
                if len(seen) >= top_k:
                    break
        
        final = sorted(seen.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results.append((img_path, final))
    
    return results


# =========================================================
# 实拍图 Top-K 检索（从文件加载模型）
# =========================================================
@torch.no_grad()
def verify_real_world_image(
    model_path,
    gallery_root,
    image_paths,
    top_k=5,
    top_n=50  # 用于去重前先取的较大 Top-N
):
    """
    实拍图 Top-K 检索（带去重，每个装备只保留最相似的一张）
    
    参数：
        model_path: embedding 模型路径
        gallery_root: gallery 图片根目录（每个子目录对应一个装备 class）
        image_paths: list, 待查询图片路径
        top_k: int, 最终输出 top-K
        top_n: int, 去重前取的 top-N
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------
    # 1. 加载模型
    # -----------------------
    checkpoint = torch.load(model_path, map_location="cpu")
    model = EmbeddingModel().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # -----------------------
    # 2. Transform
    # -----------------------
    transform = ClothingTransform(train=False)

    # -----------------------
    # 3. 构建 / 加载 gallery
    # -----------------------
    '''
    gallery_embs, gallery_labels = build_gallery(
        model,
        gallery_root,
        transform,
        device,
        cache_path=f"{model_path.replace('.pth', '_gallery.pth')}"
    )
    '''

    gallery_embs, gallery_labels = build_gallery(
        model,
        gallery_root,
        transform,
        device,
        batch_size=128,
        num_workers=8,
        cache_path=f"{model_path.replace('.pth', '_gallery.pth')}"
    )
    
    # gallery_embs = torch.nn.functional.normalize(gallery_embs, dim=1)  # 再 normalize 一次

    # -----------------------
    # 4. 查询
    # -----------------------
    for img_path in image_paths:
        img = imread_unicode(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        query = transform(img).unsqueeze(0).to(device)
        query_emb = model(query).cpu()
        query_emb = torch.nn.functional.normalize(query_emb, dim=1)

        # -----------------------
        # 5. 计算相似度 & 去重 Top-K（强制补齐）
        # -----------------------
        sims = torch.matmul(query_emb, gallery_embs.T)[0]  # [N]

        # ---------- 第一阶段：高质量候选 ----------
        vals, idxs = torch.topk(sims, min(top_n, sims.size(0)))

        seen = {}  # label -> best score
        for v, idx in zip(vals.tolist(), idxs.tolist()):
            label = gallery_labels[idx]
            if label not in seen or v > seen[label]:
                seen[label] = v

        # ---------- 第二阶段：强制补齐 ----------
        if len(seen) < top_k:
            # 按全量相似度排序（从高到低）
            all_idxs = torch.argsort(sims, descending=True)

            for idx in all_idxs.tolist():
                label = gallery_labels[idx]
                if label not in seen:
                    seen[label] = sims[idx].item()
                if len(seen) >= top_k:
                    break

        # 最终 Top-K（一定 >= top_k）
        final = sorted(seen.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # -----------------------
        # 6. 输出
        # -----------------------
        print(f"\n[Query] {os.path.basename(img_path)}")
        for i, (label, score) in enumerate(final, 1):
            print(f"Top-{i}: {label} (cos={score:.4f})")



if __name__ == "__main__":
    test_images = [
        r"S:\FFXIV_train_test\4_2.JPG",
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
        r"S:\FFXIV_train_test\5.JPG",
        r"S:\FFXIV_train_test\6.JPG",
        r"S:\FFXIV_train_test\unknown_1.JPG",
        r"S:\FFXIV_train_test\鬼师.png",
    ]

    verify_real_world_image(
        model_path="checkpoints/epoch_50_supcon.pth",
        gallery_root=r"S:\\FFXIV_train_dataset", 
        image_paths=test_images,
        top_k=10
    )