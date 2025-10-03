import cv2
import numpy as np
import os

def pad_to_same_size(img, shape):
	h, w = img.shape
	H, W = shape
	padded = np.zeros((H, W), dtype=img.dtype)
	padded[:h, :w] = img
	return padded

def main():
	# 获取当前目录下的图片文件名
	img_path = r'G:/vscode/digitalgraphcode/DIPcode/project2/4.05/4.41(a).jpg'
	template_path = r'G:/vscode/digitalgraphcode/DIPcode/project2/4.05/4.41(b).jpg'
	if not (os.path.exists(img_path) and os.path.exists(template_path)):
		print('请确保图片路径正确：', img_path, template_path)
		return
	img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
	if img is None or template is None:
		print('图片读取失败')
		return
	# 缩放图片到指定尺寸
	img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
	template = cv2.resize(template, (42, 38), interpolation=cv2.INTER_AREA)

	# 计算延拓尺寸
	H = img.shape[0] + template.shape[0] - 1
	W = img.shape[1] + template.shape[1] - 1
	max_HW = max(H, W) + 1
	print(img.shape[0], img.shape[1], template.shape[0], template.shape[1])
	# 延拓到相同大小（正方形）
	img_pad = pad_to_same_size(img, (max_HW, max_HW))
	template_pad = pad_to_same_size(template, (max_HW, max_HW))

	# 频域相关：F^-1( F(img) * conj(F(template)) )
	F_img = np.fft.fft2(img_pad)
	F_template = np.fft.fft2(template_pad)
	corr = np.fft.ifft2(F_img * np.conj(F_template))
	corr = np.abs(corr)#先求两个图片的傅里叶变换，再做乘法，最后做逆傅里叶变换，取绝对值

	# 找最大值坐标
	y, x = np.unravel_index(np.argmax(corr), corr.shape)
	print(f'Maximum correlation at (x, y): ({x}, {y})')

	# 保存相关图像
	corr_norm = cv2.normalize(corr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
	cv2.imwrite('correlation_result.png', corr_norm)
	print('相关图像已保存为 correlation_result.png')

if __name__ == '__main__':
	main()
