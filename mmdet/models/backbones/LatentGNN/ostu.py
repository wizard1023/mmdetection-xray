import torch
import cv2
import numpy as np


def normalize_tensor(input_tensor):
    """确保张量中的值在 [0, 1] 范围内"""
    min_val = input_tensor.min()
    max_val = input_tensor.max()
    normalized_tensor = (input_tensor - min_val) / (max_val - min_val + 1e-8)  # 加小值防止除零
    return normalized_tensor


def otsu_threshold(input_tensor):
    # 确保输入张量在 [0, 1] 范围内
    input_tensor = normalize_tensor(input_tensor)

    # 假设输入是 (b, c, h, w) 的张量
    b, c, h, w = input_tensor.shape

    # 创建一个输出张量
    output_tensor = torch.zeros_like(input_tensor)

    for batch in range(b):
        for channel in range(c):
            # 获取当前通道的图像并转换为 NumPy 数组
            image = input_tensor[batch, channel].cpu().numpy()

            # Otsu's 阈值法，输入图像需要为 uint8 格式
            # 将图像值从 [0, 1] 转换到 [0, 255]
            image_uint8 = (image * 255).astype(np.uint8)

            # 计算 Otsu 阈值
            if channel!=2:
                _, thresh_image = cv2.threshold(image_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif channel==2:
                _, thresh_image = cv2.threshold(image_uint8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # 将二值化的图像存储回输出张量
            output_tensor[batch, channel] = torch.tensor(thresh_image / 255.0, dtype=torch.float32)

    return output_tensor


# 使用示例
if __name__=='__main__':
    input_tensor = torch.rand(2, 3, 256, 256) * 1000  # 生成示例输入张量，可能在不同范围
    output_tensor = otsu_threshold(input_tensor)

    print(output_tensor.shape)  # 输出的张量形状
