
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

def main():
	# 读取原图像
	img = cv2.imread(r'./DIPcode/src/3.40.jpg', cv2.IMREAD_GRAYSCALE)
	import os
	print(os.getcwd())
	if img is None:
		print('Please ensure g:/vscode/digitalgraphcode/DIPcode/src/3.40.jpg exists')
		return

	# 拉普拉斯掩模（中心8，周围-1）
	laplacian_kernel = np.array([
		[-1, -1, -1],
		[-1,  8, -1],
		[-1, -1, -1]
	], dtype=np.float32)

	# 拉普拉斯滤波
	laplacian = spatial_filtering(img, laplacian_kernel)

	# c系数，可根据需要调整
	c = 1.0
	laplacian_c = multiply_images(laplacian, c)

	# 增强：g(x, y) = f(x, y) + c * [∇²f(x, y)]
	enhanced = add_images(img, laplacian_c)
	cv2.imwrite('laplacian_enhanced.png', enhanced)
	print('saved laplacian_enhanced.png')

	# 展示处理后图片
	cv2.imshow('Laplacian Enhanced', enhanced)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
