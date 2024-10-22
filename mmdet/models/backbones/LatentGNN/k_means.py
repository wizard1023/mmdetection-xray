import numpy as np
import cv2
import torch
def preprocess_image(image):
    # 将图像转换为浮点型，并进行归一化
    normalized_image = image.astype(np.float32) / 255.0

    # 调整图像大小（可根据需要调整）
    resized_image = cv2.resize(normalized_image, (500, 500))

    # 进行模糊处理，以减少噪音
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

    return blurred_image

def rgb2hsv_torch(rgb: torch.Tensor):
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.
    hsv_h /= 6.
    hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
    hsv_v = cmax
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)
def kmeans_segmentation(image, num_clusters):
    # 将图像转换为一维向量
    pixel_values = image.reshape(-1, 3).astype(np.float32)

    # 运行K-means算法
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    _, labels, centers = cv2.kmeans(pixel_values, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 将每个像素分配到最近的聚类中心
    segmented_image = centers[labels.flatten()].reshape(image.shape)

    segmented_image = np.clip(segmented_image * 255.0, 0, 255).astype(np.uint8)

    return segmented_image

# 加载图像
image = cv2.imread('/home/xray/LXM/mmdetection/demo/P00151.jpg')

# 预处理图像
processed_image = preprocess_image(image)

# 对图像进行K-means分割
num_clusters = 5  # 设置聚类簇的数量
segmented_image = kmeans_segmentation(processed_image, num_clusters)

# 显示原始图像和分割结果
# cv2.imshow('Original Image', image)
cv2.imwrite('/home/xray/LXM/mmdetection/demo/P00151_seg_1.jpg', segmented_image)


