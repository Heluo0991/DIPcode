#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
推理脚本

功能：
1. 单张图像或批量图像推理
2. 支持形态学后处理
3. 生成可视化结果和叠加图
"""

import os
import argparse
import yaml
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from tqdm import tqdm

from model import SegmentationModel
from utils import get_color_map, create_overlay, apply_morphology


class InferenceEngine:
    """推理引擎"""
    
    def __init__(self, config, checkpoint_path, device='cuda'):
        """
        参数:
            config: 配置字典
            checkpoint_path: 模型权重路径
            device: 推理设备
        """
        self.config = config
        self.device = device
        
        # 加载模型
        print("正在加载模型...")
        self.model = SegmentationModel(config)
        checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
        print(f"模型加载成功: {checkpoint_path}")
        
        # 图像预处理
        self.img_size = config['augmentation']['val']['resize']
        norm_params = config['augmentation']['val']['normalize']
        self.normalize = transforms.Normalize(
            mean=norm_params['mean'],
            std=norm_params['std']
        )
        
        # 后处理配置
        self.post_process_config = config['inference']['post_process']
        self.vis_config = config['inference']['visualization']
        
        # 颜色映射
        self.color_map = get_color_map()
    
    def preprocess(self, image_path):
        """
        预处理图像
        
        参数:
            image_path: 图像路径
        
        返回:
            image_tensor: 预处理后的tensor
            original_image: 原始图像（用于可视化）
            original_size: 原始尺寸
        """
        # 读取图像
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (W, H)
        original_image = np.array(image)
        
        # 调整大小
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # 转换为tensor
        image_tensor = transforms.ToTensor()(image)
        image_tensor = self.normalize(image_tensor)
        image_tensor = image_tensor.unsqueeze(0)  # 添加batch维度
        
        return image_tensor, original_image, original_size
    
    @torch.no_grad()
    def predict(self, image_tensor):
        """
        模型预测
        
        参数:
            image_tensor: 输入tensor (1, 3, H, W)
        
        返回:
            mask: 预测的mask (H, W)
        """
        image_tensor = image_tensor.to(self.device)
        
        # 前向传播
        output = self.model(image_tensor)
        
        # 获取预测类别
        pred = torch.argmax(output, dim=1)
        mask = pred.squeeze(0).cpu().numpy()
        
        return mask
    
    def postprocess(self, mask, original_size=None):
        """
        后处理
        
        参数:
            mask: 预测mask (H, W)
            original_size: 原始图像尺寸 (W, H)
        
        返回:
            processed_mask: 处理后的mask
        """
        # 形态学后处理
        if self.post_process_config['morphology']['enabled']:
            opening_kernel = self.post_process_config['morphology']['opening_kernel']
            closing_kernel = self.post_process_config['morphology']['closing_kernel']
            mask = apply_morphology(mask, opening_kernel, closing_kernel)
        
        # 调整回原始尺寸
        if original_size is not None:
            mask = cv2.resize(
                mask.astype(np.uint8),
                original_size,
                interpolation=cv2.INTER_NEAREST
            )
        
        return mask
    
    def visualize(self, original_image, mask, save_dir, image_name):
        """
        可视化并保存结果
        
        参数:
            original_image: 原始图像
            mask: 预测mask
            save_dir: 保存目录
            image_name: 图像名称
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存原始mask
        if self.vis_config['save_mask']:
            mask_colored = self.color_map[mask.astype(np.uint8)]
            mask_path = os.path.join(save_dir, f"{image_name}_mask.png")
            cv2.imwrite(mask_path, cv2.cvtColor(mask_colored, cv2.COLOR_RGB2BGR))
        
        # 保存叠加图
        if self.vis_config['save_overlay']:
            overlay = create_overlay(
                original_image,
                mask,
                alpha=self.vis_config['overlay_alpha']
            )
            overlay_path = os.path.join(save_dir, f"{image_name}_overlay.png")
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    def infer_single(self, image_path, save_dir):
        """
        单张图像推理
        
        参数:
            image_path: 图像路径
            save_dir: 保存目录
        """
        # 预处理
        image_tensor, original_image, original_size = self.preprocess(image_path)
        
        # 预测
        mask = self.predict(image_tensor)
        
        # 后处理
        mask = self.postprocess(mask, original_size)
        
        # 可视化
        image_name = Path(image_path).stem
        self.visualize(original_image, mask, save_dir, image_name)
        
        print(f"推理完成: {image_path}")
        return mask
    
    def infer_batch(self, input_dir, save_dir):
        """
        批量推理
        
        参数:
            input_dir: 输入目录
            save_dir: 保存目录
        """
        # 获取所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(input_dir).glob(f'*{ext}'))
            image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
        
        if len(image_files) == 0:
            print(f"在 {input_dir} 中未找到图像文件")
            return
        
        print(f"找到 {len(image_files)} 张图像")
        
        # 批量处理
        for image_path in tqdm(image_files, desc="推理进度"):
            self.infer_single(str(image_path), save_dir)
        
        print(f"\n批量推理完成! 结果保存在: {save_dir}")


def main(args):
    """主函数"""
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建推理引擎
    engine = InferenceEngine(config, args.checkpoint, device=device)
    
    # 判断输入类型
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 单张图像
        print(f"\n单张图像推理模式")
        print(f"输入图像: {input_path}")
        engine.infer_single(str(input_path), args.output)
    
    elif input_path.is_dir():
        # 批量处理
        print(f"\n批量推理模式")
        print(f"输入目录: {input_path}")
        engine.infer_batch(str(input_path), args.output)
    
    else:
        print(f"错误: 输入路径不存在 - {input_path}")
        return
    
    print(f"\n推理完成! 结果保存在: {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CelebAMask-HQ 人脸分割推理脚本')
    parser.add_argument('--input', type=str, required=True,
                        help='输入图像路径或目录')
    parser.add_argument('--output', type=str, default='results',
                        help='输出目录')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置文件路径')
    parser.add_argument('--device', type=str, default='cuda',
                        help='推理设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    main(args)
