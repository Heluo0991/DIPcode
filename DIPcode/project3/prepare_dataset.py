#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CelebAMask-HQ 数据集划分脚本

功能：
1. 读取 CelebA-HQ-to-CelebA-mapping.txt 获取图像映射关系
2. 读取 list_eval_partition.txt 获取原始 CelebA 的数据集划分
3. 根据映射关系，为 CelebAMask-HQ 生成 train_list.txt、val_list.txt 和 test_list.txt

数据集划分标识：
- 0: 训练集 (Training set)
- 1: 验证集 (Validation set)
- 2: 测试集 (Test set)
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def load_mapping(mapping_file):
    """
    加载 CelebA-HQ 到 CelebA 的映射关系
    
    参数:
        mapping_file: CelebA-HQ-to-CelebA-mapping.txt 文件路径
    
    返回:
        dict: {celebA_hq_idx: celeba_filename}
    """
    print(f"正在读取映射文件: {mapping_file}")
    mapping = {}
    
    with open(mapping_file, 'r') as f:
        # 跳过表头
        header = f.readline()
        
        for line in tqdm(f, desc="加载映射关系"):
            parts = line.strip().split()
            if len(parts) >= 3:
                hq_idx = int(parts[0])  # CelebA-HQ 索引
                orig_file = parts[2]     # 原始 CelebA 文件名
                mapping[hq_idx] = orig_file
    
    print(f"成功加载 {len(mapping)} 条映射关系")
    return mapping


def load_partition(partition_file):
    """
    加载 CelebA 数据集的划分信息
    
    参数:
        partition_file: list_eval_partition.txt 文件路径
    
    返回:
        dict: {celeba_filename: partition_id}
              partition_id: 0=train, 1=val, 2=test
    """
    print(f"正在读取划分文件: {partition_file}")
    partition = {}
    
    with open(partition_file, 'r') as f:
        for line in tqdm(f, desc="加载划分信息"):
            parts = line.strip().split()
            if len(parts) >= 2:
                filename = parts[0]
                partition_id = int(parts[1])
                partition[filename] = partition_id
    
    print(f"成功加载 {len(partition)} 条划分信息")
    return partition


def create_split_lists(mapping, partition, output_dir, verify_images=False, img_dir=None):
    """
    根据映射和划分信息，创建 CelebAMask-HQ 的数据集列表
    
    参数:
        mapping: CelebA-HQ 到 CelebA 的映射字典
        partition: CelebA 的划分信息字典
        output_dir: 输出目录
        verify_images: 是否验证图像文件存在
        img_dir: 图像目录路径（用于验证）
    """
    print("\n开始生成数据集划分列表...")
    
    # 统计信息
    split_data = defaultdict(list)
    missing_count = 0
    verified_count = 0
    
    # 遍历所有 CelebA-HQ 图像
    for hq_idx in tqdm(range(len(mapping)), desc="处理图像"):
        if hq_idx not in mapping:
            missing_count += 1
            continue
        
        # 获取对应的原始 CelebA 文件名
        celeba_filename = mapping[hq_idx]
        
        # 获取该文件的划分标识
        if celeba_filename not in partition:
            missing_count += 1
            continue
        
        partition_id = partition[celeba_filename]
        
        # CelebA-HQ 的文件名格式
        hq_filename = f"{hq_idx}.jpg"
        
        # 如果需要验证图像是否存在
        if verify_images and img_dir:
            img_path = os.path.join(img_dir, hq_filename)
            if not os.path.exists(img_path):
                print(f"警告: 图像不存在 - {img_path}")
                continue
            verified_count += 1
        
        # 根据划分标识分类
        if partition_id == 0:
            split_data['train'].append(hq_filename)
        elif partition_id == 1:
            split_data['val'].append(hq_filename)
        elif partition_id == 2:
            split_data['test'].append(hq_filename)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 写入文件
    split_names = {'train': 'train_list.txt', 'val': 'val_list.txt', 'test': 'test_list.txt'}
    
    for split_name, filename in split_names.items():
        output_file = os.path.join(output_dir, filename)
        
        # 排序以保持一致性
        split_data[split_name].sort(key=lambda x: int(x.split('.')[0]))
        
        with open(output_file, 'w') as f:
            for img_name in split_data[split_name]:
                f.write(f"{img_name}\n")
        
        print(f"已生成 {filename}: {len(split_data[split_name])} 张图像")
    
    # 打印统计信息
    print("\n" + "="*50)
    print("数据集划分统计:")
    print(f"训练集: {len(split_data['train'])} 张")
    print(f"验证集: {len(split_data['val'])} 张")
    print(f"测试集: {len(split_data['test'])} 张")
    print(f"总计:   {len(split_data['train']) + len(split_data['val']) + len(split_data['test'])} 张")
    
    if missing_count > 0:
        print(f"\n未找到映射的图像: {missing_count} 张")
    
    if verify_images:
        print(f"验证通过的图像: {verified_count} 张")
    
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description='CelebAMask-HQ 数据集划分脚本')
    parser.add_argument('--data-root', type=str, default='.',
                        help='数据集根目录 (默认: 当前目录)')
    parser.add_argument('--mapping-file', type=str, default='CelebA-HQ-to-CelebA-mapping.txt',
                        help='映射文件名 (默认: CelebA-HQ-to-CelebA-mapping.txt)')
    parser.add_argument('--partition-file', type=str, default='list_eval_partition.txt',
                        help='划分文件名 (默认: list_eval_partition.txt)')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='输出目录 (默认: 当前目录)')
    parser.add_argument('--verify-images', action='store_true',
                        help='验证图像文件是否存在')
    parser.add_argument('--img-dir', type=str, default='CelebA-HQ-img',
                        help='图像目录名 (默认: CelebA-HQ-img)')
    
    args = parser.parse_args()
    
    # 构建完整路径
    data_root = Path(args.data_root)
    mapping_file = data_root / args.mapping_file
    partition_file = data_root / args.partition_file
    output_dir = Path(args.output_dir)
    
    # 检查文件是否存在
    if not mapping_file.exists():
        raise FileNotFoundError(f"映射文件不存在: {mapping_file}")
    
    if not partition_file.exists():
        raise FileNotFoundError(f"划分文件不存在: {partition_file}")
    
    # 加载数据
    mapping = load_mapping(str(mapping_file))
    partition = load_partition(str(partition_file))
    
    # 生成划分列表
    img_dir = None
    if args.verify_images:
        img_dir = str(data_root / args.img_dir)
        if not os.path.exists(img_dir):
            print(f"警告: 图像目录不存在: {img_dir}")
            print("将继续执行但不验证图像...")
            args.verify_images = False
    
    create_split_lists(mapping, partition, str(output_dir), args.verify_images, img_dir)
    
    print("\n数据集划分完成！")
    print(f"生成的文件保存在: {output_dir}")


if __name__ == '__main__':
    main()
