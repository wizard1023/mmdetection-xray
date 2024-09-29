import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


def compute_image_gradients(image, theta_values):
    gradients = {}
    for theta in theta_values:
        # 旋转坐标轴以计算特定方向的梯度
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        # 计算x和y方向的梯度
        grad_x = np.dot(image, rotation_matrix.T)
        grad_y = np.dot(image, rotation_matrix)
        # 计算梯度幅度
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        gradients[theta] = magnitude
    return gradients


def coherent_contour_map(image, theta_values):
    # 计算图像梯度
    gradients = compute_image_gradients(image, theta_values)

    # 应用高斯平滑
    smoothed_gradients = {theta: gaussian_filter(gradient, sigma=1) for theta, gradient in gradients.items()}

    # 计算结构化张量
    tensors = {}
    for i, theta_i in enumerate(theta_values):
        for j, theta_j in enumerate(theta_values):
            if i <= j:
                tensors[(i, j)] = smoothed_gradients[theta_i] * smoothed_gradients[theta_j]

    # 生成连贯轮廓图
    coherency_map = np.sum(list(tensors.values()), axis=0)
    return coherency_map


# 读取图像
image = cv2.imread('/home/data2/lxm/datasets/SIXray_coco/val2017/P00022.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Image not found")

# 定义方向
theta_values = np.linspace(0, np.pi, 6)  # 例如，6个方向

# 生成连贯轮廓图
contour_map = coherent_contour_map(image, theta_values)

# 显示结果
cv2.imwrite('./result/contour_map.jpg',contour_map)