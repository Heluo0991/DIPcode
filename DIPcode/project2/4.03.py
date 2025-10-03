# 导入所需的库
import numpy as np  # 用于数值计算，特别是数组操作
import matplotlib.pyplot as plt  # 用于数据可视化和绘图
from PIL import Image  # Python Imaging Library，用于打开、操作和保存多种图像文件格式
import os  # 用于与操作系统交互，如此处检查文件是否存在

# 设置Matplotlib正确显示中文字符
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 指定默认字体为黑体
plt.rcParams["axes.unicode_minus"] = False  # 解决保存图像是负号'-'显示为方块的问题


def gaussian_lowpass_filter(M, N, D0, center=None):
    """
    创建一个高斯低通滤波器。

    参数:
    -----------
    M : int
        滤波器的高度
    N : int
        滤波器的宽度
    D0 : float
        截止频率 (高斯函数的标准差)
    center : tuple, optional
        滤波中心的(u0, v0)坐标。如果为None，则中心默认在(M//2, N//2)

    返回:
    --------
    H : ndarray
        一个 M x N 大小的频域高斯低通滤波器
    """
    # 如果没有指定中心，则将中心设置在频谱的中心位置
    if center is None:
        center = (M // 2, N // 2)

    u0, v0 = center

    # 创建坐标网格
    # np.arange(M) 生成 0 到 M-1 的一维数组
    # .reshape(-1, 1) 将其转换为 M行1列 的二维数组
    u = np.arange(M).reshape(-1, 1)
    v = np.arange(N).reshape(1, -1)

    # 计算每个点(u,v)到滤波器中心(u0, v0)的欧氏距离的平方
    D_squared = (u - u0) ** 2 + (v - v0) ** 2

    # 应用高斯低通滤波器公式: H(u,v) = exp(-D²(u,v) / (2·D0²))
    # D0 是截止频率，控制着滤波的程度。D0越大，滤过的低频成分越多，图像越平滑。
    H = np.exp(-D_squared / (2 * D0**2))

    return H


def apply_lowpass_filter(image, D0, filter_center=None):
    """
    对图像应用高斯低通滤波。

    参数:
    -----------
    image : ndarray
        输入的灰度图像
    D0 : float
        高斯滤波器的截止频率
    filter_center : tuple, optional
        滤波器的中心位置

    返回:
    --------
    filtered_image : ndarray
        经过低通滤波后的图像
    H : ndarray
        所使用的滤波器
    """
    # 获取图像的尺寸
    M, N = image.shape

    # --- 频域滤波的核心步骤 ---

    # 步骤 1: 对输入图像进行二维傅里叶变换(DFT)，将其从空间域转换到频域
    F = np.fft.fft2(image)

    # 步骤 2: 将零频率分量（直流分量）移动到频谱的中心
    # 这是为了方便观察和处理，因为大部分重要信息（低频分量）都集中在零频率附近
    F_shifted = np.fft.fftshift(F)

    # 步骤 3: 创建一个与图像尺寸相同的高斯低通滤波器
    H = gaussian_lowpass_filter(M, N, D0, filter_center)

    # 步骤 4: 在频域中，将图像的频谱与滤波器进行点乘
    # 这个操作会衰减高频分量，同时保留低频分量
    G_shifted = F_shifted * H

    # 步骤 5: 将频谱移回原来的位置（零频率分量回到左上角）
    G = np.fft.ifftshift(G_shifted)

    # 步骤 6: 对滤波后的频谱进行逆傅里叶变换，将其从频域转换回空间域
    filtered_image = np.fft.ifft2(G)

    # 步骤 7: 取逆变换结果的实部，并确保数据类型正确
    # 由于计算误差，逆变换结果可能包含微小的虚部，需要舍去
    filtered_image = np.real(filtered_image)

    return filtered_image, H


def visualize_results(original, filtered, H, D0):
    """
    可视化原始图像、滤波器和滤波后的结果。
    """
    # 创建一个2行3列的子图画布，设置整体大小
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # --- 第一行 ---

    # 显示原始图像
    axes[0, 0].imshow(original, cmap="gray")
    axes[0, 0].set_title("原始图像")
    axes[0, 0].axis("off")  # 关闭坐标轴

    # 显示2D高斯滤波器
    axes[0, 1].imshow(H, cmap="gray")
    axes[0, 1].set_title(f"高斯低通滤波器 (D0={D0})")
    axes[0, 1].axis("off")

    # 显示滤波后的图像
    axes[0, 2].imshow(filtered, cmap="gray")
    axes[0, 2].set_title("滤波后的图像")
    axes[0, 2].axis("off")

    # --- 第二行 ---

    # 计算并显示原始图像的对数幅度谱
    F_orig = np.fft.fftshift(np.fft.fft2(original))
    # +1是为了避免对0取对数；取对数是为了压缩动态范围，便于观察
    spectrum_orig = np.log(np.abs(F_orig) + 1)
    axes[1, 0].imshow(spectrum_orig, cmap="gray")
    axes[1, 0].set_title("原始图像频谱 (对数尺度)")
    axes[1, 0].axis("off")

    # 滤波器的三维可视化
    axes[1, 1].remove()  # 移除原有的2D子图
    ax_3d = fig.add_subplot(2, 3, 5, projection="3d")  # 在相同位置添加一个3D子图
    x = np.arange(H.shape[1])
    y = np.arange(H.shape[0])
    X, Y = np.meshgrid(x, y)  # 创建网格坐标
    ax_3d.plot_surface(X, Y, H, cmap="viridis", alpha=0.8)  # 绘制3D表面图
    ax_3d.set_title("滤波器3D可视化")
    ax_3d.set_xlabel("v (频率)")
    ax_3d.set_ylabel("u (频率)")
    ax_3d.set_zlabel("H(u,v) (幅度)")

    # 计算并显示滤波后图像的对数幅度谱
    F_filt = np.fft.fftshift(np.fft.fft2(filtered))
    spectrum_filt = np.log(np.abs(F_filt) + 1)
    axes[1, 2].imshow(spectrum_filt, cmap="gray")
    axes[1, 2].set_title("滤波后图像频谱 (对数尺度)")
    axes[1, 2].axis("off")

    plt.tight_layout()  # 自动调整子图参数，使其填充整个图像区域
    plt.show()  # 显示图像


def main():
    """
    主函数，用于演示高斯低通滤波的过程。
    """
    # 尝试加载图像 (假设图像文件位于同一目录下)
    image_path = "Figure4.18(a).jpeg"

    # 检查文件是否存在，如果不存在，则创建一个测试图像
    if os.path.exists(image_path):
        # 加载真实图像
        img = Image.open(image_path)
        # 如果不是灰度图，转换为灰度图 ('L' mode)
        if img.mode != "L":
            img = img.convert("L")
        # 将图像转换为NumPy数组，并使用浮点数类型以便计算
        image = np.array(img, dtype=np.float64)
        print(f"已加载图像: {image_path}")
    else:
        # 如果找不到图像文件，则创建一个带有一些高频噪声的合成测试图像
        print(f"警告: 图像文件 {image_path} 未找到。将使用合成的测试图像。")

        # 创建一个合成图像
        M, N = 256, 256
        x = np.linspace(0, 10 * np.pi, N)
        y = np.linspace(0, 10 * np.pi, M)
        X, Y = np.meshgrid(x, y)

        # 基础的正弦和余弦图案（低频信息）
        image = 128 + 50 * np.sin(X / 5) * np.cos(Y / 5)

        # 添加高斯噪声（高频信息）
        noise = 30 * np.random.randn(M, N)
        image = image + noise

        # 添加一些椒盐噪声（脉冲噪声，也是高频信息）
        salt_pepper = np.random.random((M, N))
        image[salt_pepper < 0.01] = 0  # 1%的像素变为黑色
        image[salt_pepper > 0.99] = 255  # 1%的像素变为白色

        # 将像素值裁剪到有效的0-255范围内
        image = np.clip(image, 0, 255)

    # 如果图像的最大值超过255，进行归一化处理
    if image.max() > 255:
        image = (image - image.min()) / (image.max() - image.min()) * 255

    # 打印图像信息
    print(f"图像尺寸: {image.shape}")
    print(f"图像像素值范围: [{image.min():.2f}, {image.max():.2f}]")

    # 演示部分：应用不同D0值的高斯低通滤波器
    print("\n正在应用高斯低通滤波器...")

    # 测试不同的截止频率
    D0_values = [10, 30, 50, 80]

    # 创建一个2行、列数为 D0_values 数量+1 的子图画布
    fig, axes = plt.subplots(2, len(D0_values) + 1, figsize=(18, 8))

    # 显示原始图像及其频谱
    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("原始图像")
    axes[0, 0].axis("off")

    F_orig = np.fft.fftshift(np.fft.fft2(image))
    spectrum_orig = np.log(np.abs(F_orig) + 1)
    axes[1, 0].imshow(spectrum_orig, cmap="gray")
    axes[1, 0].set_title("原始图像频谱")
    axes[1, 0].axis("off")

    # 遍历不同的D0值，进行滤波并显示结果
    for idx, D0 in enumerate(D0_values, 1):
        # 应用滤波器
        filtered, H = apply_lowpass_filter(image, D0)

        # 显示滤波后的图像，vmin和vmax确保不同图像的灰度范围一致
        axes[0, idx].imshow(filtered, cmap="gray", vmin=0, vmax=255)
        axes[0, idx].set_title(f"D0 = {D0}")
        axes[0, idx].axis("off")

        # 显示对应的滤波器
        axes[1, idx].imshow(H, cmap="gray")
        axes[1, idx].set_title(f"滤波器 (D0={D0})")
        axes[1, idx].axis("off")

    plt.suptitle("不同截止频率(D0)的高斯低通滤波效果对比", fontsize=14)
    plt.tight_layout()
    plt.show()

    # 使用一个较优的D0值进行详细分析
    print("\n使用 D0 = 30 进行详细分析...")
    D0_optimal = 30
    filtered_optimal, H_optimal = apply_lowpass_filter(image, D0_optimal)

    # 显示详细的可视化结果
    visualize_results(image, filtered_optimal, H_optimal, D0_optimal)

    # 测试自定义滤波器中心位置
    print("\n测试自定义滤波器中心位置...")
    M, N = image.shape
    custom_center = (M // 3, N // 3)  # 将中心设置在图像的左上部分
    H_custom = gaussian_lowpass_filter(M, N, D0=50, center=custom_center)

    # 创建子图比较不同中心的滤波器
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(H_optimal, cmap="gray")
    axes[0].set_title("中心在 (M/2, N/2) 的滤波器")
    axes[0].axis("off")

    axes[1].imshow(H_custom, cmap="gray")
    axes[1].set_title(f"中心在 {custom_center} 的滤波器")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    print("\n处理完成！")

    return image, filtered_optimal, H_optimal


# 当该脚本作为主程序运行时，执行以下代码
if __name__ == "__main__":
    original, filtered, filter_used = main()

    # 打印统计信息
    print("\n统计信息:")
    print(f"原始图像 - 均值: {original.mean():.2f}, 标准差: {original.std():.2f}")
    print(f"滤波后图像 - 均值: {filtered.mean():.2f}, 标准差: {filtered.std():.2f}")
    print(
        f"所用滤波器 - 最小值: {filter_used.min():.4f}, 最大值: {filter_used.max():.4f}"
    )
