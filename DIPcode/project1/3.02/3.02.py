import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def calculate_histogram(image):
    """
    计算灰度图像的直方图。

    参数:
        image: 代表灰度图像的 NumPy 数组。

    返回:
        一个长度为 256 的 NumPy 数组，代表直方图。
    """
    # 使用 NumPy 的 bincount 函数可以更高效地计算直方图
    hist = np.bincount(image.ravel(), minlength=256)
    return hist


def histogram_equalization(image):
    """
    对灰度图像进行全局直方图均衡化。

    参数:
        image: 代表灰度图像的 NumPy 数组。

    返回:
        一个元组，包含均衡化后的图像和变换函数。
    """
    # 步骤 1: 计算直方图
    hist = calculate_histogram(image)

    # 步骤 2: 计算累积分布函数 (CDF)
    cdf = hist.cumsum()

    # 步骤 3: 构建变换函数 (查找表)
    # 屏蔽CDF中的0值以避免除零错误
    cdf_masked = np.ma.masked_equal(cdf, 0)

    # 应用均衡化公式: T(r) = (L-1) * CDF(r) / (M*N - N_min)
    # M*N是总像素数, N_min是灰度级为0的像素数
    # 这里我们使用 cdf.max() 和 cdf_masked.min() 来进行归一化
    cdf_masked = (
        (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    )

    # 将被屏蔽的值填充为0，并转换类型
    transformation_function = np.ma.filled(cdf_masked, 0).astype("uint8")

    # 步骤 4: 应用变换函数到图像
    equalized_image = transformation_function[image]

    return equalized_image, transformation_function


# --- 主程序 ---
# 以灰度模式加载图像
try:
    image_path = "./DIPcode/project1/3.02/3.8(a).tif"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(
            f"错误：找不到图像文件 '{image_path}'。请确保文件与脚本在同一目录下。"
        )

    # --- 新增: 创建输出文件夹 ---
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # 设置matplotlib以支持中文
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # --- 1. 计算原始图像信息 ---
    original_hist = calculate_histogram(image)

    # --- 2. 执行全局直方图均衡化 ---
    global_equalized_image, transformation_function = histogram_equalization(image)
    global_equalized_hist = calculate_histogram(global_equalized_image)

    # --- 3. 执行 CLAHE (限制对比度的自适应直方图均衡化) ---
    # 这是一种改进的局部均衡化技术，通常效果更好
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(image)
    clahe_hist = calculate_histogram(clahe_image)

    # --- 4. 新增: 将每一个结果图像输出到`output`文件夹 ---

    # 4.1 保存原始图像
    cv2.imwrite(os.path.join(output_dir, "3.02.01.png"), image)

    # 4.2 保存原始直方图
    plt.figure(figsize=(6, 4))
    plt.plot(original_hist)
    plt.title("原始直方图")
    plt.xlabel("灰度级")
    plt.ylabel("像素数")
    plt.xlim([0, 255])
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "3.02.02.png"))
    plt.close()

    # 4.3 保存全局均衡化变换函数
    plt.figure(figsize=(6, 4))
    plt.plot(transformation_function)
    plt.title("全局均衡化变换函数")
    plt.xlabel("输入灰度级")
    plt.ylabel("输出灰度级")
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "3.02.03.png"))
    plt.close()

    # 4.4 保存全局均衡化图像
    cv2.imwrite(os.path.join(output_dir, "3.02.04.png"), global_equalized_image)

    # 4.5 保存全局均衡化直方图
    plt.figure(figsize=(6, 4))
    plt.plot(global_equalized_hist)
    plt.title("全局均衡化直方图")
    plt.xlabel("灰度级")
    plt.ylabel("像素数")
    plt.xlim([0, 255])
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "3.02.05.png"))
    plt.close()

    print(f"已将结果图像 3.02.01 到 3.02.05 保存到 '{output_dir}' 文件夹中。")

    # --- 5. 绘图与生成完整的报告 (保留原有功能) ---
    plt.figure(figsize=(18, 12))

    # --- 第一行: 原始图像分析 ---
    plt.subplot(3, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title("1. 原始图像")
    plt.axis("off")

    plt.subplot(3, 3, 2)
    plt.plot(original_hist)
    plt.title("2. 原始直方图")
    plt.xlabel("灰度级")
    plt.ylabel("像素数")
    plt.xlim([0, 255])

    plt.subplot(3, 3, 3)
    plt.plot(transformation_function)
    plt.title("3. 全局均衡化变换函数")
    plt.xlabel("输入灰度级")
    plt.ylabel("输出灰度级")
    plt.xlim([0, 255])
    plt.ylim([0, 255])

    # --- 第二行: 全局均衡化结果 ---
    plt.subplot(3, 3, 4)
    plt.imshow(global_equalized_image, cmap="gray")
    plt.title("4. 全局均衡化图像")
    plt.axis("off")

    plt.subplot(3, 3, 5)
    plt.plot(global_equalized_hist)
    plt.title("5. 全局均衡化直方图")
    plt.xlabel("灰度级")
    plt.ylabel("像素数")
    plt.xlim([0, 255])

    # --- 第三行: CLAHE 结果 (推荐) ---
    plt.subplot(3, 3, 7)
    plt.imshow(clahe_image, cmap="gray")
    plt.title("6. CLAHE 均衡化图像 (推荐)")
    plt.axis("off")

    plt.subplot(3, 3, 8)
    plt.plot(clahe_hist)
    plt.title("7. CLAHE 直方图")
    plt.xlabel("灰度级")
    plt.ylabel("像素数")
    plt.xlim([0, 255])

    plt.tight_layout(pad=3.0)
    plt.savefig("3.02_report.png")
    plt.show()

    print("图像处理完成，并已生成完整的报告图 '3.02_report.png'。")

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"处理过程中发生未知错误: {e}")
