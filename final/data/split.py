import os
import random
import shutil
from tqdm import tqdm

# 原始資料夾
src_root = "extracted_features"
dst_root = "split_dataset"

# 建立子資料夾
splits = ["train", "val", "test"]
for split in splits:
    os.makedirs(os.path.join(dst_root, split), exist_ok=True)

# 隨機種子可重現
random.seed(42)

# 對每個 class 進行切分
for class_name in tqdm(os.listdir(src_root), desc="Processing classes"):
    class_dir = os.path.join(src_root, class_name)
    if not os.path.isdir(class_dir):
        continue

    # 找出所有 .pt 檔
    files = [f for f in os.listdir(class_dir) if f.endswith(".pt")]
    random.shuffle(files)

    total = len(files)
    train_end = int(0.8 * total)
    val_end = int(0.9 * total)

    split_map = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }

    for split, file_list in split_map.items():
        split_class_dir = os.path.join(dst_root, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

        for fname in file_list:
            src = os.path.join(class_dir, fname)
            dst = os.path.join(split_class_dir, fname)
            shutil.copy2(src, dst)
