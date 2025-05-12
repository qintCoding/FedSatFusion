#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import glob
import rasterio

class Sen12MSDataset(Dataset):
    """Sen12MS数据集加载器"""
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        初始化Sen12MS数据集
        
        Args:
            root_dir (str): 数据集根目录
            split (str): 'train', 'val' 或 'test'
            transform (callable, optional): 数据增强转换
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # 设置数据路径
        self.s1_dir = os.path.join(root_dir, 's1')
        self.s2_dir = os.path.join(root_dir, 's2')
        self.lc_dir = os.path.join(root_dir, 'lc')
        
        # 获取所有样本ID（包括子目录中的文件）
        self.sample_ids = []
        for s1_file in glob.glob(os.path.join(self.s1_dir, '**', '*.tif'), recursive=True):
            # 从文件名中提取样本ID，例如：ROIs2017_winter_s1_94_p750.tif -> s1_94_p750
            filename = os.path.basename(s1_file)
            parts = filename.split('_')
            if len(parts) >= 4:
                sample_id = f"{parts[2]}_{parts[3]}_{parts[4].split('.')[0]}"
                self.sample_ids.append(sample_id)
        
        # 如果是训练或验证集，需要划分数据
        if split in ['train', 'val']:
            total_samples = len(self.sample_ids)
            np.random.seed(42)  # 设置随机种子确保可重复性
            indices = np.random.permutation(total_samples)
            split_idx = int(total_samples * 0.8)  # 80%用于训练
            
            if split == 'train':
                self.sample_ids = [self.sample_ids[i] for i in indices[:split_idx]]
            else:  # val
                self.sample_ids = [self.sample_ids[i] for i in indices[split_idx:]]
        
        # SAR数据的归一化参数
        self.s1_mean = [0.0, 0.0]  # VV和VH通道
        self.s1_std = [1.0, 1.0]
        
        # 光学数据的归一化参数
        self.s2_mean = [0.485, 0.456, 0.406, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        self.s2_std = [0.229, 0.224, 0.225, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    
    def find_file(self, directory, sample_id):
        """在目录及其子目录中查找文件"""
        # 从sample_id中提取关键部分，例如：s1_94_p750 -> 94_p750
        _, roi_id, patch_id = sample_id.split('_')
        search_pattern = f"*_{roi_id}_{patch_id}.tif"
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.tif') and f"_{roi_id}_{patch_id}.tif" in file:
                    return os.path.join(root, file)
        return None
    
    def load_tif(self, file_path):
        """加载.tif文件"""
        with rasterio.open(file_path) as src:
            data = src.read()  # (C, H, W)
            return data
    
    def normalize_s1(self, data):
        """归一化SAR数据"""
        data = (data - np.array(self.s1_mean)[:, None, None]) / np.array(self.s1_std)[:, None, None]
        return torch.from_numpy(data).float()
    
    def normalize_s2(self, data):
        """归一化光学数据"""
        data = (data - np.array(self.s2_mean)[:, None, None]) / np.array(self.s2_std)[:, None, None]
        return torch.from_numpy(data).float()
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        # 加载SAR数据 (S1)
        s1_path = self.find_file(self.s1_dir, sample_id)
        if s1_path is None:
            raise FileNotFoundError(f"找不到SAR数据文件: {sample_id}")
        s1_data = self.load_tif(s1_path)  # (C, H, W)
        s1_data = self.normalize_s1(s1_data)
        
        # 加载光学数据 (S2)
        s2_path = self.find_file(self.s2_dir, sample_id)
        if s2_path is None:
            raise FileNotFoundError(f"找不到光学数据文件: {sample_id}")
        s2_data = self.load_tif(s2_path)  # (C, H, W)
        s2_data = self.normalize_s2(s2_data)
        
        # 加载标签数据
        lc_path = self.find_file(self.lc_dir, sample_id)
        if lc_path is None:
            raise FileNotFoundError(f"找不到标签数据文件: {sample_id}")
        label = self.load_tif(lc_path)  # (1, H, W)
        label = torch.from_numpy(label[0]).long()  # (H, W)
        
        # 应用数据增强
        if self.transform:
            s1_data = self.transform(s1_data)
            s2_data = self.transform(s2_data)
        
        return {
            'sar': s1_data,
            'optical': s2_data,
            'label': label,
            'id': sample_id
        } 