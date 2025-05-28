import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 建立特徵提取模型
resnet = models.resnet18(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval().to(device)

# 圖像轉換器
transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_feature_sequence(folder_path):
    features = []
    filenames = sorted(os.listdir(folder_path))
    with torch.no_grad():
        for fname in filenames:
            if fname.endswith(".jpg"):
                img_path = os.path.join(folder_path, fname)
                img = Image.open(img_path).convert('RGB')
                x = transform(img).unsqueeze(0).to(device)
                feat = resnet(x)  # [1, 512, 1, 1]
                feat = feat.view(-1).cpu()  # [512]
                features.append(feat)
    return torch.stack(features)  # [16, 512]

# 根據 class 和子資料夾儲存 .pt（保留原始目錄結構）
root_dir = "processed_image"
save_root = "extracted_features"
os.makedirs(save_root, exist_ok=True)

banned_classes = ["__MACOSX", ".DS_Store"]

for class_name in tqdm(os.listdir(root_dir), desc="Processing classes"):
    if class_name in banned_classes:
        continue

    class_path = os.path.join(root_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    for subfolder in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
        subfolder_path = os.path.join(class_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        try:
            feature_tensor = extract_feature_sequence(subfolder_path)  # [16, 512]

            # 儲存到相對應的資料夾路徑
            save_subdir = os.path.join(save_root, class_name)
            os.makedirs(save_subdir, exist_ok=True)

            save_path = os.path.join(save_subdir, f"{subfolder}.pt")
            torch.save({
                "feature": feature_tensor,
                "label": class_name
            }, save_path)

        except Exception as e:
            print(f"Failed on {subfolder_path}: {e}")
