import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class EuroSATDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
        """
        初始化EuroSAT数据集
        Args:
            root_dir: 数据集根目录
            transform: 数据增强
            split: 'train' 或 'val'
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        print(f"正在加载数据集，根目录: {root_dir}")
        
        # 类别列表
        self.classes = [
            'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
            'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
            'River', 'SeaLake'
        ]
        
        # 获取所有图像路径和标签
        self.images = []
        self.labels = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            print(f"检查类别目录: {class_dir}")
            if os.path.exists(class_dir):
                image_files = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
                print(f"在 {class_name} 中找到 {len(image_files)} 张图片")
                for img_name in image_files:
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_idx)
            else:
                print(f"警告: 目录不存在 {class_dir}")
        
        print(f"总共加载了 {len(self.images)} 张图片")
        
        # 数据标准化参数
        self.mean = [0.485, 0.456, 0.406]  # RGB均值
        self.std = [0.229, 0.224, 0.225]   # RGB标准差
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        image = image.astype(np.float32) / 255.0  # 归一化到[0,1]
        
        # 转换为tensor
        image = torch.from_numpy(image).permute(2, 0, 1)  # (C, H, W)
        
        # 标准化
        image = transforms.Normalize(self.mean, self.std)(image)
        
        # 数据增强
        if self.transform:
            image = self.transform(image)
        
        return {
            'optical': image,  # (3, H, W) - RGB
            'thermal': image,  # 使用RGB作为热红外（简化处理）
            'label': torch.tensor(label, dtype=torch.long)
        } 