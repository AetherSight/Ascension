"""
清理训练目录中图片数量过少的类别目录
扫描并删除图片数量 <= 35 的目录
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict


def scan_and_cleanup(data_root, min_images=36, dry_run=True):
    """
    扫描数据目录，找出图片数量少于指定数量的目录，并删除它们
    
    Args:
        data_root: 数据根目录路径
        min_images: 最少图片数量（默认36，即删除 <= 35 的目录）
        dry_run: 如果为 True，只输出不删除（默认 True，安全模式）
    
    Returns:
        被删除/标记删除的目录列表
    """
    data_root = Path(data_root)
    
    if not data_root.exists():
        print(f"错误: 目录不存在: {data_root}")
        return []
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    # 扫描所有子目录
    low_count_dirs = []
    all_dirs_info = []
    
    print(f"扫描目录: {data_root}")
    print(f"最少图片数量要求: {min_images} (将删除 <= {min_images-1} 的目录)")
    print(f"模式: {'预览模式（不删除）' if dry_run else '删除模式'}")
    print("=" * 80)
    
    for subdir in sorted(data_root.iterdir()):
        if not subdir.is_dir():
            continue
        
        # 统计该目录下的图片数量
        image_count = 0
        for file in subdir.iterdir():
            if file.is_file() and file.suffix.lower() in image_extensions:
                image_count += 1
        
        all_dirs_info.append((subdir.name, image_count))
        
        # 如果图片数量不足，标记为待删除
        if image_count < min_images:
            low_count_dirs.append((subdir, image_count))
    
    # 输出统计信息
    print(f"\n目录统计:")
    print(f"  总目录数: {len(all_dirs_info)}")
    if all_dirs_info:
        counts = [count for _, count in all_dirs_info]
        print(f"  图片数量: 最少={min(counts)}, 最多={max(counts)}, 平均={sum(counts)/len(counts):.1f}")
    
    print(f"\n图片数量 < {min_images} 的目录数: {len(low_count_dirs)}")
    
    if low_count_dirs:
        print(f"\n待{'预览' if dry_run else '删除'}的目录列表:")
        print("-" * 80)
        
        # 按图片数量排序
        low_count_dirs.sort(key=lambda x: x[1])
        
        for idx, (dir_path, count) in enumerate(low_count_dirs, 1):
            print(f"  {idx}. {dir_path.name}")
            print(f"     路径: {dir_path}")
            print(f"     图片数量: {count}")
        
        print("-" * 80)
        
        # 执行删除操作
        if not dry_run:
            print(f"\n开始删除 {len(low_count_dirs)} 个目录...")
            deleted_count = 0
            failed_count = 0
            
            for dir_path, count in low_count_dirs:
                try:
                    shutil.rmtree(dir_path)
                    print(f"  ✓ 已删除: {dir_path.name} ({count} 张图片)")
                    deleted_count += 1
                except Exception as e:
                    print(f"  ✗ 删除失败: {dir_path.name} - {e}")
                    failed_count += 1
            
            print(f"\n删除完成:")
            print(f"  成功: {deleted_count} 个")
            print(f"  失败: {failed_count} 个")
        else:
            print(f"\n预览模式: 以上 {len(low_count_dirs)} 个目录将被删除")
            print("提示: 运行脚本时设置 dry_run=False 来执行实际删除操作")
    else:
        print(f"\n没有找到图片数量 < {min_images} 的目录")
    
    return low_count_dirs


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='清理训练目录中图片数量过少的类别目录')
    parser.add_argument('--data-root', type=str, default=r'S:\FFXIV_train_new',
                        help='数据根目录路径 (默认: S:\\FFXIV_train_new)')
    parser.add_argument('--min-images', type=int, default=70,
                        help='最少图片数量要求，低于此数量的目录将被删除 (默认: 70)')
    parser.add_argument('--execute', action='store_true',
                        help='执行删除操作（默认是预览模式，不删除）')
    
    args = parser.parse_args()
    
    # 执行清理
    low_count_dirs = scan_and_cleanup(
        data_root=args.data_root,
        min_images=args.min_images,
        dry_run=not args.execute
    )
    
    if low_count_dirs and not args.execute:
        print("\n" + "=" * 80)
        print("重要提示: 当前为预览模式，未执行实际删除操作")
        print("要执行删除，请运行: python cleanup_low_count_dirs.py --execute")
        print("=" * 80)


if __name__ == '__main__':
    main()



