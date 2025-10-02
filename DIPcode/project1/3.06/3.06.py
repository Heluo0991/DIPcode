import cv2
import numpy as np

# 3.03.py 部分
def add_images(img1, img2):
    return cv2.add(img1, img2)

def multiply_images(img1, img2_or_const):
    if isinstance(img2_or_const, (int, float)):
        return cv2.multiply(img1, np.full(img1.shape, img2_or_const, dtype=img1.dtype))
    else:
        return cv2.multiply(img1, img2_or_const)

# 3.04.py 部分
def spatial_filtering(image, kernel):
    return cv2.filter2D(image, -1, kernel)

# 高提升滤波实现
def high_boost_filter(image, A=1.0):
    """
    高提升滤波实现
    
    参数:
    image: 输入图像
    A: 高提升系数, A >= 1
    
    返回:
    高提升滤波后的图像
    """
    # 创建全1的九宫格掩码（均值滤波器），系数为1/9
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) * (1.0/9.0)
    
    # 计算模糊图像 f_bar (均值滤波)
    f_bar = spatial_filtering(image.astype(np.float32), kernel)
    
    # 计算非锐化掩模: f_mask = f(x,y) - f_bar(x,y)
    # 应用高提升滤波公式: f_hb = (A-1)*f(x,y) + f_mask(x,y)
    # 等价于: f_hb = A*f(x,y) - f_bar(x,y)
    
    # 直接使用公式 f_hb = A*f - f_bar
    f_hb = multiply_images(image.astype(np.float32), A) - f_bar
    
    # 将结果限制在合适的范围内并转换回uint8
    f_hb = np.clip(f_hb, 0, 255).astype(np.uint8)
    
    return f_hb

# 使用示例
if __name__ == "__main__":
    image = cv2.imread(r'3.43(a).png', cv2.IMREAD_GRAYSCALE)
    if image is not None:
        # Apply high-boost filtering, A=1.5
        enhanced_image = high_boost_filter(image, A=1.5)

        # Show results
        cv2.imshow('Original Image', image)
        cv2.imshow('High-Boost Filtered Image', enhanced_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save result
        cv2.imwrite('3.06result.png', enhanced_image)
    else:
        print("Failed to read image file")