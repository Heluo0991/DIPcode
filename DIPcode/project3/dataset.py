#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CelebAMask-HQ 数据集加载模块

功能：
1. 加载图像和对应的分割mask
2. 合并分散的mask文件为单个多类别mask
3. 支持数据增强
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


class CelebAMaskHQDataset(Dataset):
    """CelebAMask-HQ 数据集类"""
    
    # 类别名称映射（与mask文件名对应）
    MASK_LABELS = {
        'skin': 1, 'nose': 2, 'eye_g': 3, 'l_eye': 4, 'r_eye': 5,
        'l_brow': 6, 'r_brow': 7, 'l_ear': 8, 'r_ear': 9,
        'mouth': 10, 'u_lip': 11, 'l_lip': 12, 'hair': 13,
        'hat': 14, 'ear_r': 15, 'neck_l': 16, 'neck': 17, 'cloth': 18
    }
    
    def __init__(self, config, split='train', tiny_mode=False):
        """
        参数:
            config: 配置字典
            split: 数据集划分 ('train', 'val', 'test')
            tiny_mode: 是否使用tiny模式（用于快速验证）
        """
        self.config = config
        self.split = split
        self.tiny_mode = tiny_mode
        
        # 路径配置
        self.data_root = config['dataset']['data_root']
        self.img_dir = os.path.join(self.data_root, config['dataset']['img_dir'])
        self.mask_dir = os.path.join(self.data_root, config['dataset']['mask_dir'])
        
        # 加载图像列表
        list_file = config['dataset'][f'{split}_list']
        list_path = os.path.join(self.data_root, list_file)
        
        with open(list_path, 'r') as f:
            self.image_list = [line.strip() for line in f.readlines()]
        
        # Tiny模式：只使用部分数据
        if tiny_mode and config['tiny_mode']['enabled']:
            if split == 'train':
                self.image_list = self.image_list[:config['tiny_mode']['train_samples']]
            elif split == 'val':
                self.image_list = self.image_list[:config['tiny_mode']['val_samples']]
        
        # 数据增强配置
        self.img_size = config['augmentation'][split]['resize']
        self.setup_transforms(config['augmentation'][split])
        
        print(f"[{split.upper()}] 加载 {len(self.image_list)} 张图像")
    
    def setup_transforms(self, aug_config):
        """设置数据增强"""
        self.horizontal_flip_prob = aug_config.get('horizontal_flip', 0.0)
        self.color_jitter_params = aug_config.get('color_jitter', None)
        
        # 归一化参数
        norm_params = aug_config['normalize']
        self.normalize = transforms.Normalize(
            mean=norm_params['mean'],
            std=norm_params['std']
        )
        
        # 颜色抖动（仅用于训练）
        if self.color_jitter_params and self.split == 'train':
            self.color_jitter = transforms.ColorJitter(
                brightness=self.color_jitter_params['brightness'],
                contrast=self.color_jitter_params['contrast'],
                saturation=self.color_jitter_params['saturation'],
                hue=self.color_jitter_params['hue']
            )
        else:
            self.color_jitter = None
    
    def __len__(self):
        return len(self.image_list)
    
    def get_mask_path(self, img_idx, label_name):
        """
        获取mask文件路径
        mask文件分散在0-14共15个文件夹中，每个文件夹包含2000张图像的mask
        """
        folder_idx = img_idx // 2000
        mask_filename = f"{img_idx:05d}_{label_name}.png"
        return os.path.join(self.mask_dir, str(folder_idx), mask_filename)
    
    def load_mask(self, img_idx):
        """
        加载并合并所有类别的mask
        返回: (H, W) 的mask，每个像素值表示类别ID（0表示背景）
        """
        # 创建空白mask（512x512，背景类为0）
        mask = np.zeros((512, 512), dtype=np.uint8)
        
        # 遍历所有类别，加载对应的mask文件
        for label_name, label_id in self.MASK_LABELS.items():
            mask_path = self.get_mask_path(img_idx, label_name)
            
            if os.path.exists(mask_path):
                # 读取单通道mask
                label_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if label_mask is not None:
                    # 将非零区域标记为对应的类别ID
                    mask[label_mask > 0] = label_id
        
        return mask
    
    def __getitem__(self, idx):
        """获取单个样本"""
        # 获取图像文件名
        img_name = self.image_list[idx]
        img_idx = int(img_name.split('.')[0])
        
        # 加载图像
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # 加载mask
        mask = self.load_mask(img_idx)
        mask = Image.fromarray(mask)
        
        # 调整大小
        image = TF.resize(image, [self.img_size, self.img_size], interpolation=Image.BILINEAR)
        mask = TF.resize(mask, [self.img_size, self.img_size], interpolation=Image.NEAREST)
        
        # 数据增强（训练模式）
        if self.split == 'train':
            # 随机水平翻转
            if random.random() < self.horizontal_flip_prob:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # 颜色抖动（仅对图像）
            if self.color_jitter is not None:
                image = self.color_jitter(image)
        
        # 转换为tensor
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        
        # 归一化图像
        image = self.normalize(image)
        
        return {
            'image': image,
            'mask': mask,
            'image_name': img_name,
            'image_idx': img_idx
        }


def get_dataloader(config, split='train', tiny_mode=False):
    """
    创建数据加载器
    
    参数:
        config: 配置字典
        split: 数据集划分
        tiny_mode: 是否使用tiny模式
    
    返回:
        DataLoader对象
    """
    dataset = CelebAMaskHQDataset(config, split=split, tiny_mode=tiny_mode)
    
    # 训练集使用shuffle，验证/测试集不使用
    shuffle = (split == 'train')
    
    # 批量大小
    if split == 'train':
        batch_size = config['training']['batch_size']
    else:
        batch_size = config['inference']['batch_size']
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        drop_last=(split == 'train')  # 训练时丢弃不完整的batch
    )
    
    return dataloader


if __name__ == '__main__':
    """测试数据加载"""
    import yaml
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 启用tiny模式进行测试
    config['tiny_mode']['enabled'] = True
    
    # 创建数据加载器
    train_loader = get_dataloader(config, split='train', tiny_mode=True)
    
    print(f"\n数据加载器测试:")
    print(f"训练集批次数: {len(train_loader)}")
    
    # 加载一个batch
    batch = next(iter(train_loader))
    print(f"\n批次信息:")
    print(f"  图像形状: {batch['image'].shape}")
    print(f"  Mask形状: {batch['mask'].shape}")
    print(f"  Mask中的类别: {torch.unique(batch['mask'])}")
    print(f"  图像名称: {batch['image_name'][:5]}")
    
    print("\n数据加载测试完成！")
