import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import cv2

from lib import EmbeddingModel, imread_unicode, GalleryDataset, ClothingTransform, PatchTransform
from torch.utils.data import DataLoader
from torch.amp import autocast


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
    if cache_path and os.path.exists(cache_path):
        print(f"[Gallery] Loading cache from {cache_path}")
        data = torch.load(cache_path, map_location="cpu")
        return data["embs"], data["labels"]

    model.eval()

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

    gallery_embs = []
    gallery_labels_out = []

    for imgs, labels in tqdm(loader, desc="Building gallery"):
        imgs = imgs.to(device, non_blocking=True)

        with autocast('cuda' if device.type == "cuda" else 'cpu'):
            emb, _ = model(imgs, return_local=False)

        emb = F.normalize(emb, dim=1)
        gallery_embs.append(emb.cpu())
        gallery_labels_out.extend(labels)

    gallery_embs = torch.cat(gallery_embs, dim=0)

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


def extract_patches_from_image(img, patch_transform, model, device, num_patches=5):
    """
    Extract patch features from image for evaluation.
    Uses fixed center-based patch positions for consistent results.
    """
    patch_size = 224
    H, W = img.shape[:2]
    
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    center_h = H / 2.0
    center_w = W / 2.0
    
    if num_patches == 1:
        offset_h_ratios = np.array([0.0])
        offset_w_ratios = np.array([0.0])
    elif num_patches == 5:
        offset_h_ratios = np.array([0.0, -0.15, 0.15, 0.0, 0.0])
        offset_w_ratios = np.array([0.0, 0.0, 0.0, -0.08, 0.08])
    else:
        n = num_patches
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        radius_h = 0.15
        radius_w = 0.08
        offset_h_ratios = radius_h * np.sin(angles)
        offset_w_ratios = radius_w * np.cos(angles)
    
    offset_h = offset_h_ratios * H
    offset_w = offset_w_ratios * W
    
    patch_center_h = center_h + offset_h
    patch_center_w = center_w + offset_w
    
    top_positions = (patch_center_h - patch_size / 2.0).astype(np.int32)
    left_positions = (patch_center_w - patch_size / 2.0).astype(np.int32)
    
    top_positions = np.clip(top_positions, 0, H - patch_size)
    left_positions = np.clip(left_positions, 0, W - patch_size)
    
    patches_list = []
    for p_idx in range(num_patches):
        top = top_positions[p_idx]
        left = left_positions[p_idx]
        
        patch = img[top:top+patch_size, left:left+patch_size]
        
        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
            patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
        
        patch_tensor = patch_transform(patch)
        patches_list.append(patch_tensor)
    
    patches = torch.stack(patches_list).to(device)
    _, patch_embs = model(patches, return_local=True)
    patch_embs = F.normalize(patch_embs, dim=1)
    
    del patches
    return patch_embs.cpu()


@torch.no_grad()
def evaluate_real_world_images(
    model,
    gallery_root,
    image_paths,
    device,
    top_k=5,
    top_n=50,
    cache_path=None,
    patch_weight=0.3,
    patch_only=False
):
    """
    Evaluate real-world images against gallery.
    
    Args:
        patch_weight: Weight for patch features. If 0, patch matching is disabled.
                     Global weight will be automatically calculated as 1 - patch_weight.
        patch_only: If True, treat input image as a single patch (no extraction needed).
                    The image itself is used as patch feature for matching.
    """
    if patch_only:
        global_weight = 0.0
        patch_weight = 1.0
    else:
        global_weight = 1.0 - patch_weight
    
    use_patch_match = patch_weight > 0
    num_patches = 5  # Fixed number of patches when extracting from full image
    
    model.eval()
    transform = ClothingTransform(train=False)
    patch_transform = PatchTransform(return_tensor=True) if use_patch_match else None
    
    gallery_embs, gallery_labels = build_gallery(
        model,
        gallery_root,
        transform,
        device,
        batch_size=128,
        num_workers=4,
        cache_path=cache_path
    )
    
    results = []
    for img_path in image_paths:
        if not os.path.exists(img_path):
            continue
            
        img = imread_unicode(img_path)
        if img is None:
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        sims_global = None
        if not patch_only:
            query = transform(img).unsqueeze(0).to(device)
            query_emb, _ = model(query, return_local=False)
            query_emb = query_emb.cpu()
            query_emb = F.normalize(query_emb, dim=1)
            sims_global = torch.matmul(query_emb, gallery_embs.T)[0]
        
        sims_patch = None
        sims_patch_per_patch = None
        if use_patch_match:
            if patch_only:
                patch_tensor = patch_transform(img).unsqueeze(0).to(device)
                _, patch_emb = model(patch_tensor, return_local=True)
                patch_emb = F.normalize(patch_emb, dim=1).cpu()
                sims_patch = torch.matmul(patch_emb, gallery_embs.T)[0]
                sims_patch_per_patch = sims_patch.unsqueeze(0)
            else:
                patch_embs = extract_patches_from_image(
                    img, patch_transform, model, device, num_patches
                )
                
                sims_patch_list = []
                for patch_emb in patch_embs:
                    sims_patch = torch.matmul(patch_emb.unsqueeze(0), gallery_embs.T)[0]
                    sims_patch_list.append(sims_patch)
                
                sims_patch_stack = torch.stack(sims_patch_list)
                sims_patch_per_patch = sims_patch_stack
                
                if num_patches > 2:
                    max_vals, _ = sims_patch_stack.max(dim=0)
                    min_vals, _ = sims_patch_stack.min(dim=0)
                    sims_patch = (sims_patch_stack.sum(dim=0) - max_vals - min_vals) / (num_patches - 2)
                else:
                    sims_patch = sims_patch_stack.mean(dim=0)
        
        if patch_only:
            sims = sims_patch
        elif use_patch_match:
            sims = global_weight * sims_global + patch_weight * sims_patch
        else:
            sims = sims_global
        
        vals, idxs = torch.topk(sims, min(top_n, sims.size(0)))
        
        seen = {}
        for v, idx in zip(vals.tolist(), idxs.tolist()):
            label = gallery_labels[idx]
            global_score = sims_global[idx].item() if sims_global is not None else None
            patch_score = sims_patch[idx].item() if sims_patch is not None else None
            patch_scores = sims_patch_per_patch[:, idx].tolist() if sims_patch_per_patch is not None else None
            if label not in seen or v > seen[label][0]:
                seen[label] = (v, global_score, patch_score, patch_scores)
        
        if len(seen) < top_k:
            all_idxs = torch.argsort(sims, descending=True)
            for idx in all_idxs.tolist():
                label = gallery_labels[idx]
                if label not in seen:
                    global_score = sims_global[idx].item() if sims_global is not None else None
                    patch_score = sims_patch[idx].item() if sims_patch is not None else None
                    patch_scores = sims_patch_per_patch[:, idx].tolist() if sims_patch_per_patch is not None else None
                    seen[label] = (sims[idx].item(), global_score, patch_score, patch_scores)
                if len(seen) >= top_k:
                    break
        
        final = sorted(seen.items(), key=lambda x: x[1][0], reverse=True)[:top_k]
        results.append((img_path, final))
    
    return results


@torch.no_grad()
def verify_real_world_image(
    model_path,
    gallery_root,
    image_paths,
    top_k=5,
    top_n=50,
    patch_weight=0.3,
    patch_only=False
):
    """
    Load model from file and evaluate real-world images.
    
    Args:
        model_path: Path to model checkpoint
        gallery_root: Gallery root directory
        image_paths: List of query image paths
        top_k: Return top-K results
        top_n: Top-N candidates before deduplication
        patch_weight: Weight for patch features. If 0, patch matching is disabled.
                     Global weight will be automatically calculated as 1 - patch_weight.
        patch_only: If True, treat input image as a single patch (no extraction needed).
                    The image itself is used as patch feature for matching.
    """
    if patch_only:
        global_weight = 0.0
        patch_weight = 1.0
    else:
        global_weight = 1.0 - patch_weight
    
    use_patch_match = patch_weight > 0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location="cpu")
    model = EmbeddingModel().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    cache_path = f"{model_path.replace('.pth', '_gallery.pth')}"
    results = evaluate_real_world_images(
        model=model,
        gallery_root=gallery_root,
        image_paths=image_paths,
        device=device,
        top_k=top_k,
        top_n=top_n,
        cache_path=cache_path,
        patch_weight=patch_weight,
        patch_only=patch_only
    )

    for img_path, final in results:
        print(f"\n[Query] {os.path.basename(img_path)}")
        for i, (label, score_info) in enumerate(final, 1):
            final_score, global_score, patch_score, patch_scores = score_info
            if patch_only and patch_score is not None and patch_scores is not None:
                print(f"Top-{i}: {label} | Patch: {patch_score:.4f} | Final: {final_score:.4f} (patch-only)")
            elif use_patch_match and patch_score is not None and patch_scores is not None:
                if len(patch_scores) > 2:
                    patch_scores_sorted = sorted(patch_scores)
                    max_score = patch_scores_sorted[-1]
                    min_score = patch_scores_sorted[0]
                    remaining_scores = patch_scores_sorted[1:-1]
                    
                    patch_strs = []
                    max_found = False
                    min_found = False
                    for score in patch_scores:
                        if not max_found and abs(score - max_score) < 1e-6:
                            patch_strs.append(f"{score:.3f}(max)")
                            max_found = True
                        elif not min_found and abs(score - min_score) < 1e-6:
                            patch_strs.append(f"{score:.3f}(min)")
                            min_found = True
                        else:
                            patch_strs.append(f"{score:.3f}")
                    
                    remaining_str = "+".join([f"{s:.3f}" for s in remaining_scores])
                    print(f"Top-{i}: {label} | Patch: [{', '.join(patch_strs)}] → ({remaining_str})/{len(remaining_scores)}={patch_score:.4f} | Final: {final_score:.4f} = {global_weight:.1f}×{global_score:.4f} + {patch_weight:.1f}×{patch_score:.4f}")
                else:
                    patch_str = "+".join([f"{s:.3f}" for s in patch_scores])
                    print(f"Top-{i}: {label} | Patch: ({patch_str})/{len(patch_scores)}={patch_score:.4f} | Final: {final_score:.4f} = {global_weight:.1f}×{global_score:.4f} + {patch_weight:.1f}×{patch_score:.4f}")
            else:
                print(f"Top-{i}: {label} (cos={final_score:.4f})")
    
    return results



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
        r"S:\FFXIV_train_test\玉韦亚瓦塔强袭短衣.png",
        r"S:\FFXIV_train_test\小象服.webp",
    ]

    verify_real_world_image(
        model_path="checkpoints_0.0.3/epoch_30_supcon.pth",
        gallery_root=r"S:\\FFXIV_train_dataset", 
        image_paths=test_images,
        top_k=10,
        patch_weight=0.3,
        patch_only=False
    )