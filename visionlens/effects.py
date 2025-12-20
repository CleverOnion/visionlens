"""
特效处理模块
实现 Alpha 混合和多挂件融合
"""

import cv2
import numpy as np
from typing import List, Tuple


def alpha_blend(background: np.ndarray,
                foreground: np.ndarray,
                alpha: np.ndarray) -> np.ndarray:
    """
    Alpha 混合实现自然边缘过渡
    
    Args:
        background: 背景图像（BGR）
        foreground: 前景图像（BGR）
        alpha: Alpha 通道（0-255）
        
    Returns:
        混合后的图像（BGR）
    """
    # 确保输入格式正确
    background_f = background.astype(np.float32)
    foreground_f = foreground.astype(np.float32)
    alpha_f = alpha.astype(np.float32) / 255.0
    
    # 扩展 Alpha 通道维度以匹配图像通道数
    if len(background.shape) == 3:
        alpha_3d = np.expand_dims(alpha_f, axis=2)
    else:
        alpha_3d = alpha_f
    
    # Alpha 混合公式: result = alpha * foreground + (1 - alpha) * background
    # 只在 alpha > 0 的区域进行混合，避免全透明区域影响
    mask = alpha_3d > 0.01
    blended = np.where(mask,
                      alpha_3d * foreground_f + (1 - alpha_3d) * background_f,
                      background_f)
    
    return blended.astype(np.uint8)


def blend_multiple_accessories(background: np.ndarray,
                               accessories: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    融合多个饰品到背景图像（性能优化版本）

    Args:
        background: 背景图像（BGR）
        accessories: 饰品列表，每个元素为 (foreground_image, alpha_channel)

    Returns:
        融合后的图像（BGR）
    """
    if not accessories:
        return background

    # 性能优化：避免不必要的copy操作，直接在背景上操作
    result = background.astype(np.float32)

    # 按顺序渲染每个饰品
    for foreground, alpha in accessories:
        # 性能优化：提前验证输入，减少后续计算
        if (foreground is None or alpha is None or
            foreground.shape[:2] != result.shape[:2] or
            alpha.shape[:2] != result.shape[:2]):
            continue

        # 性能优化：快速检查alpha有效性
        alpha_max = alpha.max()
        if alpha_max == 0:
            continue

        # 性能优化：预处理alpha和foreground
        alpha_norm = alpha.astype(np.float32) / 255.0

        # 性能优化：使用更高效的alpha扩展方法
        if len(alpha_norm.shape) == 2:
            alpha_3d = alpha_norm[:, :, np.newaxis]  # 比expand_dims更高效
        else:
            alpha_3d = alpha_norm

        # 性能优化：确保foreground格式正确
        if foreground.dtype != np.float32:
            foreground_f = foreground.astype(np.float32)
        else:
            foreground_f = foreground

        # 性能优化：使用更高效的混合计算
        # 预先计算alpha和1-alpha的乘积
        alpha_mult = alpha_3d
        one_minus_alpha = 1.0 - alpha_mult

        # 向量化混合计算
        blended = alpha_mult * foreground_f + one_minus_alpha * result

        # 性能优化：使用更高效的mask应用
        mask = alpha_3d > 0.001
        result = np.where(mask, blended, result)

    return result.astype(np.uint8)


def apply_alpha_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    应用 Alpha 遮罩到图像
    
    Args:
        image: 输入图像（BGR 或 BGRA）
        mask: Alpha 遮罩（单通道，0-255）
        
    Returns:
        应用遮罩后的图像（BGRA）
    """
    if len(image.shape) == 2:
        # 灰度图转 BGR
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    if image.shape[2] == 3:
        # BGR 转 BGRA
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    # 应用 Alpha 通道
    image[:, :, 3] = mask
    
    return image

