import cv2


def process_image_with_opencv(input_path, output_path):
    """
    使用OpenCV将图片转换为688x688的TIF格式文件。

    参数:
    input_path (str): 输入图片的路径。
    output_path (str): 输出TIF文件的路径。
    """
    # 读取图像
    img = cv2.imread(input_path)
    if img is not None:
        # 统一转换为688x688大小
        resized_img = cv2.resize(img, (688, 688))
        # 保存为TIF格式
        cv2.imwrite(output_path, resized_img)
        print(f"成功将 {input_path} 转换为 {output_path}")
    else:
        print(f"无法读取图像: {input_path}")


# 使用示例
process_image_with_opencv("Figure4.18(a).jpeg", "Figure4.18(a).tif")
