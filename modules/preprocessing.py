# -*- coding: utf-8 -*-
"""
=============================================================================
图像预处理模块 - 手写英文图片优化
=============================================================================
功能: 对上传的手写图片进行预处理，提升 EasyOCR 对手写体的识别准确率

处理流程:
  原始图片 → 灰度化 → 降噪 → 对比度增强 → 二值化 → 倾斜矫正 → 裁剪 → 输出

各步骤目的:
  灰度化:      减少颜色噪声干扰，简化后续处理
  降噪:        去除扫描/拍照产生的噪点、纸张纹理
  对比度增强:  使浅色笔迹更清晰（CLAHE 算法）
  二值化:      转换为黑白图像，突出文字轮廓（自适应阈值）
  倾斜矫正:    修正拍照/书写时的角度偏差
  裁剪:        去除白边，聚焦文字区域
=============================================================================
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import math

from config import PREPROCESSING


def grayscale(image: np.ndarray) -> np.ndarray:
    """
    灰度化 - 将彩色图像转为灰度图

    如果图片已经是灰度图则直接返回。
    使用加权转换公式: Gray = 0.299R + 0.587G + 0.114B
    """
    if len(image.shape) == 2:
        return image  # 已经是灰度图
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(image: np.ndarray) -> np.ndarray:
    """
    降噪处理 - 去除噪点和纸张纹理

    双层降噪策略:
    1. 先使用高斯模糊去除轻微噪点
    2. 再用 Non-Local Means 去噪保边缘
    """
    h = PREPROCESSING["denoise_strength"]
    # 高斯模糊（轻量，去除椒盐噪点）
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    # NLMeans 去噪（保留文字边缘）
    denoised = cv2.fastNlMeansDenoising(blurred, None, h, 7, 21)
    return denoised


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    对比度增强 - 使用 CLAHE 算法

    CLAHE (Contrast Limited Adaptive Histogram Equalization):
    自适应直方图均衡化，在局部区域增强对比度而不放大噪声。
    特别适合浅色铅笔字迹、光照不均的拍照图片。
    """
    clip_limit = PREPROCESSING["clahe_clip_limit"]
    grid_size = PREPROCESSING["clahe_grid_size"]
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)


def binarize(image: np.ndarray) -> np.ndarray:
    """
    二值化 - 自适应阈值转为纯黑白

    使用自适应高斯阈值:
    - 每个像素的阈值由其周围 block_size 区域的加权均值决定
    - 比全局阈值 (Otsu) 更适合光照不均的图片
    - 文字区域为黑色(0)，背景为白色(255)
    """
    block_size = PREPROCESSING["binary_block_size"]
    C = PREPROCESSING["binary_C"]

    # block_size 必须是奇数
    if block_size % 2 == 0:
        block_size += 1

    binary = cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # 反转: 文字白色便于后续处理
        block_size,
        C
    )

    # ---- 形态学操作：去除小噪点，连接断笔 ----
    kernel = np.ones((2, 2), np.uint8)
    # 开运算: 先腐蚀后膨胀，去除小白点噪声
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # 闭运算: 先膨胀后腐蚀，连接断开的笔画
    kernel_close = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    return binary


def deskew(image: np.ndarray) -> np.ndarray:
    """
    倾斜矫正 - 检测并纠正图片的倾斜角度

    方法: 基于文本行的投影轮廓
    1. 对多个角度计算投影轮廓的方差
    2. 方差最大的角度 = 文字行最清晰的角度 = 倾斜角
    3. 旋转图片进行矫正

    限制: 矫正角度在 ±max_angle 范围内
    """
    max_angle = PREPROCESSING["deskew_max_angle"]

    # 确保是二值图（文字为白色前景）
    if len(image.shape) > 2:
        gray = grayscale(image)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # 检查是否需要反转（确保文字是白色/前景）
        white_pixels = np.sum(image > 127)
        black_pixels = np.sum(image <= 127)
        if black_pixels > white_pixels:
            binary = image  # 文字已经是前景
        else:
            binary = cv2.bitwise_not(image)

    # 计算最佳旋转角度：投影轮廓方差最大
    best_angle = 0.0
    best_variance = 0.0

    coords = np.column_stack(np.where(binary > 0))

    if len(coords) == 0:
        return image  # 无内容，无需矫正

    for angle in np.arange(-max_angle, max_angle + 1, 0.5):
        # 旋转坐标点
        theta = np.radians(angle)
        rotated_coords = np.dot(coords, [[np.cos(theta), -np.sin(theta)],
                                          [np.sin(theta), np.cos(theta)]])
        # 投影到水平方向的方差
        hist, _ = np.histogram(rotated_coords[:, 0], bins=50)
        variance = np.var(hist)

        if variance > best_variance:
            best_variance = variance
            best_angle = angle

    # 执行旋转矫正
    if abs(best_angle) > 0.5:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        # 旋转后填充白色
        if len(image.shape) == 2:
            rotated = cv2.warpAffine(
                image, rotation_matrix, (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255
            )
        else:
            rotated = cv2.warpAffine(
                image, rotation_matrix, (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255)
            )
        return rotated

    return image


def remove_borders(image: np.ndarray, padding: int = 10) -> np.ndarray:
    """
    裁剪白边 - 去除图片周围的空白区域

    通过查找文字轮廓的边界框来定位文字区域，
    然后裁剪到文字区域 + 少量内边距。
    """
    # 确保是二值图（文字为前景）
    if len(image.shape) > 2:
        gray = grayscale(image)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 查找非零像素的边界
    coords = cv2.findNonZero(binary)
    if coords is None:
        return image  # 无内容

    x, y, w, h = cv2.boundingRect(coords)

    # 加入内边距
    img_h, img_w = image.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)

    return image[y1:y2, x1:x2]


def preprocess_image(
    image: np.ndarray,
    enable_deskew: bool = True,
    enable_border_removal: bool = True,
    return_intermediates: bool = False
) -> np.ndarray:
    """
    完整预处理流水线

    Args:
        image: 输入图片 (numpy array, BGR 或 Gray)
        enable_deskew: 是否启用倾斜矫正
        enable_border_removal: 是否裁剪白边
        return_intermediates: 是否返回中间步骤结果

    Returns:
        预处理后的图片 (numpy array)，或 (final_image, intermediates_dict)

    处理顺序: 灰度 → 降噪 → 对比度增强 → 二值化 → 倾斜矫正 → 裁剪
    """
    intermediates = {}

    # Step 1: 灰度化
    gray = grayscale(image)
    intermediates["grayscale"] = gray

    # Step 2: 降噪
    denoised = denoise(gray)
    intermediates["denoised"] = denoised

    # Step 3: 对比度增强
    enhanced = enhance_contrast(denoised)
    intermediates["enhanced"] = enhanced

    # Step 4: 二值化
    binary = binarize(enhanced)
    intermediates["binary"] = binary

    # Step 5: 倾斜矫正（可选）
    if enable_deskew:
        binary = deskew(binary)
        intermediates["deskewed"] = binary

    # Step 6: 裁剪白边（可选）
    if enable_border_removal:
        binary = remove_borders(binary)
        intermediates["cropped"] = binary

    if return_intermediates:
        return binary, intermediates

    return binary


def load_and_preprocess(
    image_path: str,
    **kwargs
) -> np.ndarray:
    """
    从文件路径加载图片并预处理

    Args:
        image_path: 图片文件路径
        **kwargs: 传递给 preprocess_image 的参数

    Returns:
        预处理后的图片数组
    """
    # 使用 imread 读取（支持中文路径）
    image = cv2.imdecode(
        np.fromfile(image_path, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")

    return preprocess_image(image, **kwargs)


def visualize_preprocessing(image: np.ndarray) -> np.ndarray:
    """
    可视化预处理各步骤 - 调试用

    返回水平拼接的各步骤图片，方便对比效果。
    """
    _, intermediates = preprocess_image(image, return_intermediates=True)

    # 收集各步骤图片
    steps = []
    labels = []

    for step_name, step_img in intermediates.items():
        # 确保是 3 通道用于显示
        if len(step_img.shape) == 2:
            step_display = cv2.cvtColor(step_img, cv2.COLOR_GRAY2BGR)
        else:
            step_display = step_img.copy()

        # 调整到统一高度
        target_height = 300
        h, w = step_display.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        step_display = cv2.resize(step_display, (new_w, target_height))

        steps.append(step_display)
        labels.append(step_name)

    # 水平拼接所有步骤
    result = np.hstack(steps)
    return result
