#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型定义模块

使用 segmentation_models_pytorch 库快速构建带预训练骨干的 U-Net 模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class SegmentationModel(nn.Module):
    """分割模型包装类"""
    
    def __init__(self, config):
        """
        参数:
            config: 配置字典
        """
        super(SegmentationModel, self).__init__()
        
        self.config = config
        model_config = config['model']
        
        # 从配置读取参数
        architecture = model_config['architecture']
        encoder_name = model_config['encoder']
        encoder_weights = model_config['encoder_weights']
        in_channels = model_config['in_channels']
        classes = model_config['classes']
        
        # 使用 segmentation_models_pytorch 创建模型
        if architecture.lower() == 'unet':
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=None  # 不使用激活函数，后续使用softmax
            )
        elif architecture.lower() == 'unetplusplus':
            self.model = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=None
            )
        elif architecture.lower() == 'deeplabv3plus':
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=None
            )
        else:
            raise ValueError(f"不支持的架构: {architecture}")
        
        print(f"模型创建成功: {architecture} with {encoder_name}")
        print(f"  输入通道: {in_channels}, 输出类别: {classes}")
        print(f"  预训练权重: {encoder_weights}")
    
    def forward(self, x):
        """前向传播"""
        return self.model(x)


class CombinedLoss(nn.Module):
    """组合损失函数：交叉熵 + Dice Loss"""
    
    def __init__(self, ce_weight=1.0, dice_weight=0.5, num_classes=19, ignore_index=255):
        """
        参数:
            ce_weight: 交叉熵损失权重
            dice_weight: Dice损失权重
            num_classes: 类别数量
            ignore_index: 忽略的标签索引
        """
        super(CombinedLoss, self).__init__()
        
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.num_classes = num_classes
        
        # 交叉熵损失
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def dice_loss(self, pred, target, smooth=1e-6):
        """
        计算多类别 Dice Loss
        
        参数:
            pred: 预测logits (B, C, H, W)
            target: 真实标签 (B, H, W)
            smooth: 平滑项
        """
        # 将预测转换为概率
        pred = F.softmax(pred, dim=1)
        
        # 将target转换为one-hot编码
        target_one_hot = F.one_hot(target, num_classes=self.num_classes)  # (B, H, W, C)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # 计算Dice系数
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        
        # 返回Dice Loss（1 - Dice）
        return 1.0 - dice.mean()
    
    def forward(self, pred, target):
        """
        计算组合损失
        
        参数:
            pred: 预测logits (B, C, H, W)
            target: 真实标签 (B, H, W)
        """
        # 交叉熵损失
        ce = self.ce_loss(pred, target)
        
        # Dice损失
        dice = self.dice_loss(pred, target)
        
        # 组合损失
        total_loss = self.ce_weight * ce + self.dice_weight * dice
        
        return total_loss, ce, dice


def get_model(config, device='cuda'):
    """
    获取模型实例
    
    参数:
        config: 配置字典
        device: 设备
    
    返回:
        model: 模型实例
        criterion: 损失函数
    """
    # 创建模型
    model = SegmentationModel(config)
    model = model.to(device)
    
    # 创建损失函数
    loss_weights = config['training']['loss_weights']
    criterion = CombinedLoss(
        ce_weight=loss_weights['ce_weight'],
        dice_weight=loss_weights['dice_weight'],
        num_classes=config['dataset']['num_classes']
    )
    
    return model, criterion


def count_parameters(model):
    """统计模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    return total_params, trainable_params


if __name__ == '__main__':
    """测试模型"""
    import yaml
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("创建模型...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, criterion = get_model(config, device=device)
    
    # 统计参数
    count_parameters(model)
    
    # 测试前向传播
    print("\n测试前向传播...")
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 512, 512).to(device)
    dummy_target = torch.randint(0, 19, (batch_size, 512, 512)).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    
    # 测试损失函数
    print("\n测试损失函数...")
    loss, ce, dice = criterion(output, dummy_target)
    print(f"总损失: {loss.item():.4f}")
    print(f"交叉熵损失: {ce.item():.4f}")
    print(f"Dice损失: {dice.item():.4f}")
    
    print("\n模型测试完成！")
