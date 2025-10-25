#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具函数模块

功能：
1. 评估指标计算（mIoU, Pixel Accuracy, Dice）
2. 可视化功能
3. 辅助函数
"""

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os


class MetricTracker:
    """指标跟踪器"""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """重置所有指标"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def update(self, pred, target):
        """
        更新混淆矩阵
        
        参数:
            pred: 预测mask (H, W) 或 (B, H, W)
            target: 真实mask (H, W) 或 (B, H, W)
        """
        pred = pred.flatten()
        target = target.flatten()
        
        # 只统计有效的类别（0到num_classes-1）
        mask = (target >= 0) & (target < self.num_classes)
        
        # 更新混淆矩阵
        indices = self.num_classes * target[mask].astype(int) + pred[mask].astype(int)
        self.confusion_matrix += np.bincount(
            indices, minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
    
    def get_miou(self, ignore_background=True):
        """
        计算平均IoU
        
        参数:
            ignore_background: 是否忽略背景类
        """
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusion_matrix)
        union = (
            self.confusion_matrix.sum(axis=1) +  # TP + FN
            self.confusion_matrix.sum(axis=0) -  # TP + FP
            intersection  # 减去重复计算的TP
        )
        
        # 避免除零
        iou = np.zeros(self.num_classes)
        valid = union > 0
        iou[valid] = intersection[valid] / union[valid]
        
        # 是否忽略背景
        if ignore_background and self.num_classes > 1:
            return np.mean(iou[1:])
        else:
            return np.mean(iou)
    
    def get_pixel_accuracy(self):
        """计算像素准确率"""
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        return correct / total if total > 0 else 0.0
    
    def get_dice(self, ignore_background=True):
        """
        计算Dice系数
        Dice = 2 * TP / (2 * TP + FP + FN)
        """
        intersection = np.diag(self.confusion_matrix)
        pred_sum = self.confusion_matrix.sum(axis=0)
        target_sum = self.confusion_matrix.sum(axis=1)
        
        dice = np.zeros(self.num_classes)
        valid = (pred_sum + target_sum) > 0
        dice[valid] = 2 * intersection[valid] / (pred_sum[valid] + target_sum[valid])
        
        if ignore_background and self.num_classes > 1:
            return np.mean(dice[1:])
        else:
            return np.mean(dice)
    
    def get_class_iou(self):
        """获取每个类别的IoU"""
        intersection = np.diag(self.confusion_matrix)
        union = (
            self.confusion_matrix.sum(axis=1) +
            self.confusion_matrix.sum(axis=0) -
            intersection
        )
        
        iou = np.zeros(self.num_classes)
        valid = union > 0
        iou[valid] = intersection[valid] / union[valid]
        return iou


def get_color_map(num_classes=19):
    """
    生成颜色映射表
    
    返回:
        color_map: (num_classes+1, 3) 的RGB颜色数组
    """
    # 为19个类别 + 背景生成不同的颜色
    colors = [
        [0, 0, 0],         # 0: background (黑色)
        [255, 200, 200],   # 1: skin (浅粉色)
        [255, 150, 150],   # 2: nose (粉色)
        [200, 200, 255],   # 3: eye_g (浅蓝色)
        [100, 100, 255],   # 4: l_eye (蓝色)
        [150, 150, 255],   # 5: r_eye (中蓝色)
        [255, 255, 100],   # 6: l_brow (黄色)
        [255, 255, 150],   # 7: r_brow (浅黄色)
        [255, 150, 100],   # 8: l_ear (橙色)
        [255, 180, 130],   # 9: r_ear (浅橙色)
        [255, 100, 100],   # 10: mouth (红色)
        [255, 50, 50],     # 11: u_lip (深红色)
        [255, 0, 0],       # 12: l_lip (纯红色)
        [150, 100, 50],    # 13: hair (棕色)
        [100, 50, 150],    # 14: hat (紫色)
        [255, 215, 0],     # 15: ear_r (金色)
        [0, 255, 255],     # 16: neck_l (青色)
        [200, 150, 100],   # 17: neck (浅棕色)
        [100, 150, 200],   # 18: cloth (浅灰蓝)
    ]
    
    return np.array(colors, dtype=np.uint8)


def visualize_segmentation(image, mask, pred=None, save_path=None, class_names=None):
    """
    可视化分割结果
    
    参数:
        image: 原始图像 (H, W, 3) 或 (3, H, W)
        mask: 真实mask (H, W)
        pred: 预测mask (H, W), 可选
        save_path: 保存路径
        class_names: 类别名称列表
    """
    # 转换图像格式
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    if image.shape[0] == 3:  # (3, H, W) -> (H, W, 3)
        image = np.transpose(image, (1, 2, 0))
    
    # 反归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    # 转换mask格式
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    
    # 生成颜色映射
    color_map = get_color_map()
    
    # 创建彩色mask
    mask_colored = color_map[mask.astype(np.uint8)]
    
    # 确定子图数量
    n_plots = 3 if pred is not None else 2
    
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    
    if n_plots == 2:
        axes = [axes[0], axes[1]]
    
    # 显示原始图像
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 显示真实mask
    axes[1].imshow(mask_colored)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # 显示预测mask
    if pred is not None:
        pred_colored = color_map[pred.astype(np.uint8)]
        axes[2].imshow(pred_colored)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_overlay(image, mask, alpha=0.5):
    """
    创建分割结果的叠加图
    
    参数:
        image: 原始图像 (H, W, 3), numpy array, 范围[0, 255]
        mask: 分割mask (H, W), numpy array
        alpha: 叠加透明度（此参数在当前实现中不使用）
    
    返回:
        overlay: 叠加图像 (H, W, 3)
    """
    # 确保图像格式正确
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # 创建overlay图像，初始化为黑色
    overlay = np.zeros_like(image)
    
    # 创建前景掩码（mask != 0 表示非背景区域）
    foreground_mask = mask > 0
    
    # 在前景区域显示原图，背景区域保持黑色
    overlay[foreground_mask] = image[foreground_mask]
    
    return overlay


def apply_morphology(mask, opening_kernel=3, closing_kernel=5):
    """
    应用形态学后处理
    
    参数:
        mask: 分割mask (H, W), numpy array
        opening_kernel: 开运算核大小
        closing_kernel: 闭运算核大小
    
    返回:
        processed_mask: 处理后的mask
    """
    # 为每个类别分别处理
    processed_mask = np.zeros_like(mask)
    unique_labels = np.unique(mask)
    
    for label in unique_labels:
        if label == 0:  # 跳过背景
            continue
        
        # 提取单个类别
        binary_mask = (mask == label).astype(np.uint8)
        
        # 开运算：去除小噪点
        if opening_kernel > 0:
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_kernel, opening_kernel))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
        
        # 闭运算：填充小空洞
        if closing_kernel > 0:
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # 合并回结果
        processed_mask[binary_mask > 0] = label
    
    return processed_mask


class AverageMeter:
    """计算并存储平均值和当前值"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, save_path, is_best=False):
    """
    保存模型检查点
    
    参数:
        state: 包含模型状态的字典
        save_path: 保存路径
        is_best: 是否是最佳模型
    """
    torch.save(state, save_path)
    
    if is_best:
        best_path = save_path.replace('.pth', '_best.pth')
        torch.save(state, best_path)
        print(f"最佳模型已保存: {best_path}")


def load_checkpoint(model, checkpoint_path, device='cuda'):
    """
    加载模型检查点
    
    参数:
        model: 模型
        checkpoint_path: 检查点路径
        device: 设备
    
    返回:
        epoch: 训练轮次
        best_miou: 最佳mIoU
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_miou = checkpoint.get('best_miou', 0.0)
    
    print(f"已加载检查点: {checkpoint_path}")
    print(f"  Epoch: {epoch}, Best mIoU: {best_miou:.4f}")
    
    return epoch, best_miou


if __name__ == '__main__':
    """测试工具函数"""
    print("测试评估指标...")
    
    # 创建模拟数据
    num_classes = 19
    pred = np.random.randint(0, num_classes, size=(512, 512))
    target = np.random.randint(0, num_classes, size=(512, 512))
    
    # 测试指标计算
    tracker = MetricTracker(num_classes)
    tracker.update(pred, target)
    
    print(f"mIoU: {tracker.get_miou():.4f}")
    print(f"Pixel Accuracy: {tracker.get_pixel_accuracy():.4f}")
    print(f"Dice: {tracker.get_dice():.4f}")
    
    print("\n测试颜色映射...")
    color_map = get_color_map()
    print(f"颜色映射形状: {color_map.shape}")
    
    print("\n工具函数测试完成！")
