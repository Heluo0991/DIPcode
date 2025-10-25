#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练脚本

功能：
1. 使用迁移学习训练U-Net模型
2. 支持TensorBoard日志记录
3. 支持早停和模型保存
4. 支持Tiny模式快速验证
"""

import os
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from dataset import get_dataloader
from model import get_model, count_parameters
from utils import MetricTracker, AverageMeter, save_checkpoint


def train_one_epoch(
    model, dataloader, criterion, optimizer, device, epoch, writer, config
):
    """训练一个epoch"""
    model.train()

    loss_meter = AverageMeter()
    ce_meter = AverageMeter()
    dice_meter = AverageMeter()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        # 前向传播
        outputs = model(images)
        loss, ce_loss, dice_loss = criterion(outputs, masks)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新指标
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        ce_meter.update(ce_loss.item(), batch_size)
        dice_meter.update(dice_loss.item(), batch_size)

        # 更新进度条
        pbar.set_postfix(
            {
                "loss": f"{loss_meter.avg:.4f}",
                "ce": f"{ce_meter.avg:.4f}",
                "dice": f"{dice_meter.avg:.4f}",
            }
        )

        # TensorBoard记录
        if config["logging"]["tensorboard"]:
            global_step = epoch * len(dataloader) + batch_idx
            if batch_idx % config["logging"]["print_frequency"] == 0:
                writer.add_scalar("Train/Loss", loss.item(), global_step)
                writer.add_scalar("Train/CE_Loss", ce_loss.item(), global_step)
                writer.add_scalar("Train/Dice_Loss", dice_loss.item(), global_step)

    return loss_meter.avg, ce_meter.avg, dice_meter.avg


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, num_classes):
    """验证"""
    model.eval()

    loss_meter = AverageMeter()
    ce_meter = AverageMeter()
    dice_meter = AverageMeter()

    # 创建指标跟踪器
    metric_tracker = MetricTracker(num_classes)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")

    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        # 前向传播
        outputs = model(images)
        loss, ce_loss, dice_loss = criterion(outputs, masks)

        # 获取预测结果
        preds = torch.argmax(outputs, dim=1)

        # 更新损失
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        ce_meter.update(ce_loss.item(), batch_size)
        dice_meter.update(dice_loss.item(), batch_size)

        # 更新指标
        metric_tracker.update(preds.cpu().numpy(), masks.cpu().numpy())

        # 更新进度条
        pbar.set_postfix(
            {
                "loss": f"{loss_meter.avg:.4f}",
                "mIoU": f"{metric_tracker.get_miou():.4f}",
            }
        )

    # 计算最终指标
    miou = metric_tracker.get_miou(ignore_background=True)
    pixel_acc = metric_tracker.get_pixel_accuracy()
    dice = metric_tracker.get_dice(ignore_background=True)

    return loss_meter.avg, miou, pixel_acc, dice


def main(args):
    """主训练函数"""

    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 命令行参数覆盖配置
    if args.tiny:
        config["tiny_mode"]["enabled"] = True
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr

    # 设置设备
    device = torch.device(
        config["training"]["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"使用设备: {device}")

    # 创建保存目录
    save_dir = config["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # 创建日志目录
    if config["logging"]["tensorboard"]:
        log_dir = config["logging"]["log_dir"]
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None

    # 创建数据加载器
    print("\n" + "=" * 50)
    print("加载数据集...")
    train_loader = get_dataloader(config, split="train", tiny_mode=args.tiny)
    val_loader = get_dataloader(config, split="val", tiny_mode=args.tiny)
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")

    # 创建模型
    print("\n" + "=" * 50)
    print("创建模型...")
    model, criterion = get_model(config, device=device)
    count_parameters(model)

    # 创建优化器
    print("\n" + "=" * 50)
    print("创建优化器...")
    if config["training"]["optimizer"].lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
    elif config["training"]["optimizer"].lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            momentum=0.9,
            weight_decay=config["training"]["weight_decay"],
        )
    else:
        raise ValueError(f"不支持的优化器: {config['training']['optimizer']}")

    print(f"优化器: {config['training']['optimizer']}")
    print(f"学习率: {config['training']['learning_rate']}")

    # 创建学习率调度器
    scheduler_type = config["training"]["scheduler"]
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["scheduler_params"]["T_max"],
            eta_min=config["training"]["scheduler_params"]["eta_min"],
        )
    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )
    else:
        scheduler = None

    # 训练循环
    print("\n" + "=" * 50)
    print("开始训练...")
    print("=" * 50)

    num_epochs = config["training"]["num_epochs"]
    best_miou = 0.0
    patience_counter = 0
    patience = config["training"]["early_stopping"]["patience"]

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 50)

        # 训练
        train_loss, train_ce, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer, config
        )

        # 验证
        val_loss, val_miou, val_pixel_acc, val_dice = validate(
            model,
            val_loader,
            criterion,
            device,
            epoch,
            config["dataset"]["num_classes"],
        )

        # 更新学习率
        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(val_miou)
            else:
                scheduler.step()

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]["lr"]

        # 打印结果
        print(
            f"\n训练 - Loss: {train_loss:.4f}, CE: {train_ce:.4f}, Dice: {train_dice:.4f}"
        )
        print(
            f"验证 - Loss: {val_loss:.4f}, mIoU: {val_miou:.4f}, Pixel Acc: {val_pixel_acc:.4f}, Dice: {val_dice:.4f}"
        )
        print(f"学习率: {current_lr:.6f}")

        # TensorBoard记录
        if writer is not None:
            writer.add_scalar("Epoch/Train_Loss", train_loss, epoch)
            writer.add_scalar("Epoch/Val_Loss", val_loss, epoch)
            writer.add_scalar("Epoch/Val_mIoU", val_miou, epoch)
            writer.add_scalar("Epoch/Val_Pixel_Acc", val_pixel_acc, epoch)
            writer.add_scalar("Epoch/Val_Dice", val_dice, epoch)
            writer.add_scalar("Epoch/Learning_Rate", current_lr, epoch)

        # 保存检查点
        is_best = val_miou > best_miou
        if is_best:
            best_miou = val_miou
            patience_counter = 0
            print(f"*** 新的最佳模型! mIoU: {best_miou:.4f} ***")
        else:
            patience_counter += 1

        # 保存模型
        if config["training"]["save_best_only"]:
            if is_best:
                save_path = os.path.join(save_dir, f"best_model.pth")
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_miou": best_miou,
                        "config": config,
                    },
                    save_path,
                    is_best=True,
                )
        else:
            if epoch % config["training"]["save_frequency"] == 0:
                save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_miou": best_miou,
                        "config": config,
                    },
                    save_path,
                    is_best=is_best,
                )

        # 早停
        if patience_counter >= patience:
            print(f"\n早停触发! {patience} 个epoch没有改善")
            break

    # 训练结束
    print("\n" + "=" * 50)
    print("训练完成!")
    print(f"最佳 mIoU: {best_miou:.4f}")
    print("=" * 50)

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CelebAMask-HQ 人脸分割训练脚本")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="配置文件路径"
    )
    parser.add_argument("--tiny", action="store_true", help="使用Tiny模式（快速验证）")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="批量大小（覆盖配置文件）"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="训练轮数（覆盖配置文件）"
    )
    parser.add_argument("--lr", type=float, default=None, help="学习率（覆盖配置文件）")

    args = parser.parse_args()

    main(args)
