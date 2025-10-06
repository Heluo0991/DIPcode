import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- 全局设置: 用于解决 Matplotlib 中文显示问题 ---
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 'SimHei' 是黑体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号'-'显示为方块的问题

# --- 核心功能函数 (复用) ---


def gaussian_lowpass_filter(M, N, D0, center=None):
    """
    生成一个 M x N 的高斯低通滤波器。

    参数:
    M, N: 滤波器尺寸
    D0: 截止频率 (高斯函数的标准差)
    center: 滤波器中心位置, 默认为 (M/2, N/2)

    返回:
    H: M x N 的高斯低通滤波器
    """
    if center is None:
        center = (M // 2, N // 2)
    u0, v0 = center

    u = np.arange(M).reshape(-1, 1)
    v = np.arange(N).reshape(1, -1)

    D_squared = (u - u0) ** 2 + (v - v0) ** 2
    H = np.exp(-D_squared / (2 * D0**2))

    return H


def apply_filter_in_freq_domain(image, H):
    """
    在频域中对图像应用一个给定的滤波器H。
    """
    # 1. 计算图像的傅里葉变换并中心化
    F_shifted = np.fft.fftshift(np.fft.fft2(image))

    # 2. 应用滤波器
    G_shifted = F_shifted * H

    # 3. 反中心化并进行傅里葉逆变换
    G = np.fft.ifftshift(G_shifted)
    filtered_image = np.real(np.fft.ifft2(G))

    return filtered_image


def normalize_for_display(image):
    """
    将图像的灰度值线性拉伸到 [0, 255] 以便显示。
    """
    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        # 线性拉伸
        normalized = (image - img_min) / (img_max - img_min) * 255
    else:
        # 如果图像是纯色
        normalized = image
    return normalized


# --- 项目实现函数 ---


def run_project_04_03(original_image, D0):
    """
    执行 Project 04-03 的任务：高斯低通滤波，并演示自定义中心位置。
    """
    print(f"\n--- [开始] Project 04-03: 高斯低通滤波 (D0 = {D0}) ---")

    M, N = original_image.shape

    # --- Part 1: 标准中心滤波 ---
    # 1. 生成标准的高斯低通滤波器 (中心在 M/2, N/2)
    H_lowpass_centered = gaussian_lowpass_filter(M, N, D0)

    # 2. 应用滤波器
    lowpass_image_centered = apply_filter_in_freq_domain(
        original_image, H_lowpass_centered
    )

    # 3. 结果可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Project 04-03: 标准中心高斯低通滤波 (D0={D0})", fontsize=16)

    axes[0].imshow(original_image, cmap="gray")
    axes[0].set_title("原始图像 (Fig. 4.18a)")
    axes[0].axis("off")

    axes[1].imshow(H_lowpass_centered, cmap="gray")
    axes[1].set_title("中心化滤波器")
    axes[1].axis("off")

    axes[2].imshow(lowpass_image_centered, cmap="gray")
    axes[2].set_title("滤波结果 (类似 Fig. 4.18c)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    # --- Part 2: 演示自定义中心位置功能 ---
    print("\n[补充演示] 展示指定滤波器中心位置的功能...")

    # 1. 定义一个偏离中心的坐标
    custom_center = (M // 4, N // 4)

    # 2. 生成偏心滤波器
    H_lowpass_off_center = gaussian_lowpass_filter(M, N, D0, center=custom_center)

    # 3. 应用偏心滤波器
    lowpass_image_off_center = apply_filter_in_freq_domain(
        original_image, H_lowpass_off_center
    )

    # 4. 可视化对比
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("滤波器中心位置对结果的影响对比", fontsize=16)

    # 显示两个滤波器
    axes[0, 0].imshow(H_lowpass_centered, cmap="gray")
    axes[0, 0].set_title(f"标准中心滤波器\nCenter:({M//2}, {N//2})")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(H_lowpass_off_center, cmap="gray")
    axes[0, 1].set_title(f"偏心滤波器\nCenter:{custom_center}")
    axes[0, 1].axis("off")

    # 显示两个滤波结果
    axes[1, 0].imshow(lowpass_image_centered, cmap="gray")
    axes[1, 0].set_title("标准中心滤波结果")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(lowpass_image_off_center, cmap="gray")
    axes[1, 1].set_title("偏心滤波结果")
    axes[1, 1].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    print("--- [完成] Project 04-03 ---")

    # 返回标准中心滤波的结果，用于 04-04
    return lowpass_image_centered


def run_project_04_04(original_image, lowpass_image, target_image):
    """
    执行 Project 04-04 的任务：通过低通图像实现高通滤波。
    """
    print("\n--- [开始] Project 04-04: 使用低通图像进行高通滤波 ---")

    # --- (a) 部分：从原图中减去低通滤波图像 ---
    print("\n[Part a] 执行图像相减...")
    highpass_by_sub = original_image - lowpass_image

    # 为了显示，需要将结果归一化
    highpass_display = normalize_for_display(highpass_by_sub)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(lowpass_image, cmap="gray")
    plt.title("用于相减的低通图像")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(highpass_display, cmap="gray")
    plt.title("高通结果 (原图 - 低通图)")
    plt.axis("off")
    plt.suptitle("Project 04-04 (a): 图像相减实现高通", fontsize=16)
    plt.show()

    # --- (b) 部分：调整 D0 使结果接近目标图像 ---
    print("\n[Part b] 寻找最优 D0 以匹配目标图像...")

    D0_range = np.arange(5, 100, 5)
    best_D0 = -1
    min_error = float("inf")
    best_highpass_result = None

    for D0_test in D0_range:
        H_lp_test = gaussian_lowpass_filter(
            original_image.shape[0], original_image.shape[1], D0_test
        )
        lp_image_test = apply_filter_in_freq_domain(original_image, H_lp_test)
        hp_image_test = original_image - lp_image_test
        error = np.mean(
            (normalize_for_display(hp_image_test) - normalize_for_display(target_image))
            ** 2
        )

        if error < min_error:
            min_error = error
            best_D0 = D0_test
            best_highpass_result = hp_image_test

    print(f"搜索完成! 最优 D0 = {best_D0} (均方误差最小)")

    # 结果可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Project 04-04 (b): 调整 D0 以匹配目标", fontsize=16)

    axes[0].imshow(target_image, cmap="gray")
    axes[0].set_title("目标图像 (Fig. 4.26c)")
    axes[0].axis("off")

    axes[1].imshow(highpass_display, cmap="gray")
    axes[1].set_title(f"初始结果 (D0={D0_INITIAL})")
    axes[1].axis("off")

    axes[2].imshow(normalize_for_display(best_highpass_result), cmap="gray")
    axes[2].set_title(f"最佳匹配结果 (D0={best_D0})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    print("--- [完成] Project 04-04 ---")


# --- 主程序入口 ---


def main():
    """
    主函数，按顺序执行所有项目任务。
    """
    # --- 准备工作：加载图像 ---
    print("正在加载所需图像...")
    original_image_path = "Figure4.18(a).tif"
    target_image_path = "Figure4.26(c).tif"

    original_image = np.array(
        Image.open(original_image_path).convert("L"), dtype=np.float64
    )
    target_image = np.array(
        Image.open(target_image_path).convert("L"), dtype=np.float64
    )
    print("图像加载完成。")

    # --- 执行 Project 04-03 ---
    global D0_INITIAL
    D0_INITIAL = 30
    lowpass_image_result = run_project_04_03(original_image, D0=D0_INITIAL)

    # --- 执行 Project 04-04 ---
    run_project_04_04(original_image, lowpass_image_result, target_image)

    print("\n所有项目任务已执行完毕。")


if __name__ == "__main__":
    main()
