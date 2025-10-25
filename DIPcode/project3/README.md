# CelebAMask-HQ 人脸分割项目

基于深度学习的高精度人脸语义分割系统，使用 CelebAMask-HQ 数据集训练 U-Net 模型。

## 📋 项目简介

本项目实现了一个高性能的人脸分割模型，能够对人脸图像进行精确的像素级语义分割，识别19个不同的人脸部位和配饰类别。

### 主要特性

- ✅ **迁移学习**：使用 ImageNet 预训练的 ResNet34 作为编码器骨干
- ✅ **多类别分割**：支持19个人脸部位类别的精确分割
- ✅ **数据增强**：随机翻转、颜色抖动等增强策略
- ✅ **组合损失**：交叉熵 + Dice Loss
- ✅ **形态学后处理**：自动去除噪点和填充空洞
- ✅ **TensorBoard支持**：实时监控训练过程
- ✅ **Tiny模式**：快速验证代码逻辑

### 分割类别（19类）

1. skin (皮肤)
2. nose (鼻子)
3. eye_g (眼镜)
4. l_eye (左眼)
5. r_eye (右眼)
6. l_brow (左眉)
7. r_brow (右眉)
8. l_ear (左耳)
9. r_ear (右耳)
10. mouth (嘴巴内部)
11. u_lip (上嘴唇)
12. l_lip (下嘴唇)
13. hair (头发)
14. hat (帽子)
15. ear_r (耳环)
16. neck_l (项链)
17. neck (脖子)
18. cloth (衣服)

## 🚀 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境（推荐）
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

确保 CelebAMask-HQ 数据集已下载并按以下结构组织：

```
CelebAMask-HQ/
├── CelebA-HQ-img/           # 30,000张高分辨率图像
├── CelebAMask-HQ-mask-anno/ # 分割mask标注
├── CelebA-HQ-to-CelebA-mapping.txt
└── list_eval_partition.txt
```

运行数据集划分脚本：

```bash
python prepare_dataset.py --data-root . --output-dir .
```

这将生成：
- `train_list.txt` (24,183张图像)
- `val_list.txt` (2,993张图像)
- `test_list.txt` (2,824张图像)

### 3. 训练模型

#### 快速验证（Tiny模式）

使用少量数据快速测试代码：

```bash
python train.py --tiny --epochs 2
```

#### 完整训练

```bash
# 使用默认配置
python train.py

# 自定义参数
python train.py --batch-size 16 --epochs 50 --lr 0.001
```

训练过程中的指标：
- 训练损失、验证损失
- mIoU（平均交并比）
- 像素准确率
- Dice系数

### 4. 监控训练

使用 TensorBoard 可视化训练过程：

```bash
tensorboard --logdir logs
```

然后在浏览器中访问 `http://localhost:6006`

### 5. 模型推理

#### 单张图像推理

```bash
python inference.py \
    --input path/to/image.jpg \
    --checkpoint checkpoints/best_model.pth \
    --output results
```

#### 批量推理

```bash
python inference.py \
    --input path/to/images/ \
    --checkpoint checkpoints/best_model.pth \
    --output results
```

推理结果包括：
- `*_mask.png`：彩色分割mask
- `*_overlay.png`：原图与mask的叠加图

## 📁 项目结构

```
CelebAMask-HQ/
├── config.yaml              # 配置文件
├── prepare_dataset.py       # 数据集划分脚本
├── dataset.py              # 数据加载模块
├── model.py                # 模型定义
├── utils.py                # 工具函数
├── train.py                # 训练脚本
├── inference.py            # 推理脚本
├── requirements.txt        # 依赖列表
├── README.md              # 项目文档
├── checkpoints/           # 模型保存目录
├── logs/                  # TensorBoard日志
└── results/               # 推理结果
```

## ⚙️ 配置说明

所有超参数在 `config.yaml` 中配置，主要配置项：

### 模型配置
```yaml
model:
  architecture: "unet"        # 模型架构
  encoder: "resnet34"         # 编码器骨干
  encoder_weights: "imagenet" # 预训练权重
```

### 训练配置
```yaml
training:
  batch_size: 16              # 批量大小
  num_epochs: 50              # 训练轮数
  learning_rate: 0.001        # 学习率
  optimizer: "adam"           # 优化器
  scheduler: "cosine"         # 学习率调度
```

### 数据增强
```yaml
augmentation:
  train:
    horizontal_flip: 0.5      # 水平翻转概率
    color_jitter:
      brightness: 0.2
      contrast: 0.2
```

## 📊 性能指标

### 目标性能
- **mIoU**: > 75%
- **像素准确率**: > 90%
- **推理速度**: < 200ms/图 (RTX 4070, 512x512)

### 评估指标
- **mIoU (Mean IoU)**: 平均交并比，主要评估指标
- **Pixel Accuracy**: 像素级准确率
- **Dice Coefficient**: Dice系数

## 🔧 高级用法

### 修改模型架构

在 `config.yaml` 中更改骨干网络：

```yaml
model:
  encoder: "resnet50"  # 可选: resnet34, resnet50, efficientnet-b0等
```

### 调整损失函数权重

```yaml
training:
  loss_weights:
    ce_weight: 1.0      # 交叉熵损失权重
    dice_weight: 0.5    # Dice损失权重
```

### 形态学后处理参数

```yaml
inference:
  post_process:
    morphology:
      enabled: true
      opening_kernel: 3   # 开运算核大小
      closing_kernel: 5   # 闭运算核大小
```

## 🐛 常见问题

### 1. CUDA Out of Memory

减小批量大小：
```bash
python train.py --batch-size 8
```

### 2. 数据加载慢

调整工作线程数（在 `config.yaml` 中）：
```yaml
training:
  num_workers: 4  # 根据CPU核心数调整
```

### 3. 快速测试代码

使用 Tiny 模式：
```bash
python train.py --tiny --epochs 2
```

## 📝 引用

如果使用本项目或 CelebAMask-HQ 数据集，请引用：

```bibtex
@article{CelebAMask-HQ,
  title={MaskGAN: Towards Diverse and Interactive Facial Image Manipulation},
  author={Cheng-Han Lee and Ziwei Liu and Lingyun Wu and Ping Luo},
  journal={Technical Report},
  year={2019}
}
```

## 📄 许可

本项目仅用于非商业研究和教育目的。CelebAMask-HQ 数据集的版权归原作者所有。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题，请通过 GitHub Issues 联系。

---

**开发环境**: Python 3.8+, PyTorch 1.12+, CUDA 11.3+
