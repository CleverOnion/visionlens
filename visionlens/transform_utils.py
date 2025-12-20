"""
3D 变换工具模块
实现饰品的缩放、旋转和透视变换
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from .face_detector import FaceDetector


def calculate_glasses_transform(landmarks: np.ndarray, 
                                face_detector: FaceDetector,
                                glasses_width: int,
                                glasses_height: int) -> Optional[np.ndarray]:
    """
    计算眼镜的变换矩阵
    
    Args:
        landmarks: 468 个关键点
        face_detector: FaceDetector 实例
        glasses_width: 眼镜图片宽度
        glasses_height: 眼镜图片高度
        
    Returns:
        3x3 仿射变换矩阵，如果计算失败返回 None
    """
    if landmarks is None or len(landmarks) < 468:
        return None
        
    key_points = face_detector.get_key_points(landmarks)
    
    # 检查关键点是否有效
    if not key_points or len(key_points) == 0:
        return None
    
    # 获取两眼中心点
    left_eye_center = (key_points['left_eye_left'] + key_points['left_eye_right']) / 2
    right_eye_center = (key_points['right_eye_left'] + key_points['right_eye_right']) / 2
    
    # 计算两眼中心点
    eye_center = (left_eye_center + right_eye_center) / 2
    
    # 眼镜应该稍微向上移动，放在眼睛上方
    # 使用鼻梁点作为参考，眼镜应该在眼睛和鼻梁之间
    nose_bridge = key_points['nose_bridge']
    # 眼镜位置在眼睛中心向上移动一点（约眼睛到鼻梁距离的20%）
    # 注意：在图像坐标中，y 轴向下，所以向上移动需要减去 y 值
    eye_to_nose = nose_bridge - eye_center
    # 向上移动：减去 eye_to_nose 的 y 分量
    glasses_position = eye_center.copy()
    glasses_position[1] = eye_center[1] - abs(eye_to_nose[1]) * 0.2  # 向上移动
    
    # 计算两眼间距离
    eye_distance = np.linalg.norm(right_eye_center - left_eye_center)
    
    # 计算缩放比例（基于两眼距离）
    # 假设标准眼镜宽度约为两眼距离的 1.2 倍
    # 对于大图片（>500px），使用更大的缩放因子
    target_width = eye_distance * 1.2
    if glasses_width > 500:
        # 对于大图片，假设图片中的眼镜只占图片的一部分（比如中心30%区域）
        # 这样缩放比例会更合理
        effective_width = glasses_width * 0.3
        scale = target_width / effective_width
    else:
        scale = target_width / glasses_width
    
    # 整体再缩小一点（例如 0.8 倍），让画面中的眼镜更精致
    scale *= 0.8
    
    # 限制缩放范围，避免过小或过大
    scale = np.clip(scale, 0.05, 5.0)
    
    # 计算旋转角度（Roll）
    eye_vector = right_eye_center - left_eye_center
    angle = np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))
    
    # 计算头部姿态
    pitch, yaw, roll = face_detector.calculate_head_pose(landmarks)
    
    # 根据 Yaw 角度调整透视效果
    # Yaw 越大，透视形变越明显
    yaw_degrees = np.degrees(yaw)
    perspective_factor = 1.0 + abs(yaw_degrees) / 90.0 * 0.3  # 最大 30% 的透视形变
    
    # 图像中心点
    center_x = glasses_width / 2
    center_y = glasses_height / 2
    
    # 计算旋转角度（弧度）
    angle_rad = np.radians(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # 缩放因子
    scale_x = scale * perspective_factor
    scale_y = scale
    
    # 计算围绕中心点旋转和缩放后的中心点位置
    # 对于 cv2.warpAffine，变换矩阵是 [a b tx; c d ty]
    # 对于点 (cx, cy)，变换后是 (a*cx + b*cy + tx, c*cx + d*cy + ty)
    # 我们希望这个结果等于目标位置 (glasses_position[0], glasses_position[1])
    
    a = scale_x * cos_a
    b = -scale_x * sin_a
    c = scale_y * sin_a
    d = scale_y * cos_a
    
    # 计算平移量，使得变换后的中心点移动到目标位置
    tx = glasses_position[0] - (a * center_x + b * center_y)
    ty = glasses_position[1] - (c * center_x + d * center_y)
    
    # 构建 2x3 仿射变换矩阵（用于 cv2.warpAffine）
    # 格式：[a b tx]
    #       [c d ty]
    M = np.array([
        [a, b, tx],
        [c, d, ty]
    ])
    
    return M


def calculate_beard_transform(landmarks: np.ndarray,
                               face_detector: FaceDetector,
                               beard_width: int,
                               beard_height: int) -> Optional[np.ndarray]:
    """
    计算胡子的变换矩阵
    
    Args:
        landmarks: 468 个关键点
        face_detector: FaceDetector 实例
        beard_width: 胡子图片宽度
        beard_height: 胡子图片高度
        
    Returns:
        3x3 仿射变换矩阵，如果计算失败返回 None
    """
    if landmarks is None or len(landmarks) < 468:
        return None
        
    key_points = face_detector.get_key_points(landmarks)
    
    # 检查关键点是否有效
    if not key_points or len(key_points) == 0:
        return None
    
    # 获取下巴和鼻梁点
    chin = key_points['chin']
    nose_bridge = key_points['nose_bridge']
    
    # 计算下巴到鼻梁的距离
    nose_to_chin_distance = np.linalg.norm(chin - nose_bridge)
    
    # 计算缩放比例
    # 假设标准胡子高度约为鼻梁到下巴距离的 0.6 倍
    target_height = nose_to_chin_distance * 0.6
    
    # 对于大图片（>500px），使用更大的缩放因子
    if beard_height > 500:
        # 对于大图片，假设图片中的胡子只占图片的一部分（比如中心30%区域）
        effective_height = beard_height * 0.3
        scale = target_height / effective_height
    else:
        scale = target_height / beard_height
    
    # 计算宽度（基于脸颊宽度）
    # 使用左右眼外侧点估算脸颊宽度
    left_eye_left = key_points['left_eye_left']
    right_eye_right = key_points['right_eye_right']
    face_width = np.linalg.norm(right_eye_right - left_eye_left) * 1.5
    
    if beard_width > 500:
        effective_width = beard_width * 0.3
        width_scale = face_width / effective_width
    else:
        width_scale = face_width / beard_width
    
    # 使用较小的缩放比例以保持比例
    scale = min(scale, width_scale)
    
    # 限制缩放范围，避免过大或过小
    scale = np.clip(scale, 0.05, 5.0)
    
    # 计算旋转角度（基于头部 Roll）
    pitch, yaw, roll = face_detector.calculate_head_pose(landmarks)
    angle = np.degrees(roll)
    
    # 根据 Yaw 调整透视
    yaw_degrees = np.degrees(yaw)
    perspective_factor = 1.0 + abs(yaw_degrees) / 90.0 * 0.2
    
    # 图像中心点
    center_x = beard_width / 2
    center_y = beard_height / 2
    
    # 计算旋转角度（弧度）
    angle_rad = np.radians(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # 缩放因子
    scale_x = scale * perspective_factor
    scale_y = scale
    
    # 计算围绕中心点旋转和缩放后的中心点位置
    # 对于 cv2.warpAffine，变换矩阵是 [a b tx; c d ty]
    # 对于点 (cx, cy)，变换后是 (a*cx + b*cy + tx, c*cx + d*cy + ty)
    
    a = scale_x * cos_a
    b = -scale_x * sin_a
    c = scale_y * sin_a
    d = scale_y * cos_a
    
    # 4. 平移回并移动到目标位置（定位到下巴上方）
    # 胡子应该位于下巴和嘴唇之间
    beard_position = chin + (nose_bridge - chin) * 0.3
    
    # 计算平移量，使得变换后的中心点移动到目标位置
    tx = beard_position[0] - (a * center_x + b * center_y)
    ty = beard_position[1] - (c * center_x + d * center_y)
    
    # 构建 2x3 仿射变换矩阵（用于 cv2.warpAffine）
    # 格式：[a b tx]
    #       [c d ty]
    M = np.array([
        [a, b, tx],
        [c, d, ty]
    ])
    
    return M


def apply_transform(image: np.ndarray,
                    transform_matrix: np.ndarray,
                    output_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    应用仿射变换到图像（性能优化版本）

    Args:
        image: 输入图像（BGRA 格式，包含 Alpha 通道）
        transform_matrix: 2x3 仿射变换矩阵
        output_shape: 输出图像尺寸 (height, width)

    Returns:
        (transformed_image, transformed_alpha) 变换后的图像和 Alpha 通道
    """
    # 性能优化：提前验证输入
    if image is None or transform_matrix is None:
        return None, None

    # 性能优化：预先计算输出尺寸，避免重复计算
    output_width, output_height = output_shape[1], output_shape[0]

    if len(image.shape) == 3 and image.shape[2] == 4:
        # 分离颜色和 Alpha 通道
        bgr = image[:, :, :3]
        alpha = image[:, :, 3]

        # 性能优化：使用线性插值平衡质量和性能
        interpolation_flags = cv2.INTER_LINEAR

        # 分别变换颜色和 Alpha 通道
        transformed_bgr = cv2.warpAffine(
            bgr, transform_matrix, (output_width, output_height),
            flags=interpolation_flags,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        transformed_alpha = cv2.warpAffine(
            alpha, transform_matrix, (output_width, output_height),
            flags=interpolation_flags,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        # 性能优化：避免不必要的合并操作，直接返回分离结果
        # 调用者需要的格式已经调整
        transformed_image = np.dstack([transformed_bgr, transformed_alpha])
        return transformed_image, transformed_alpha
    else:
        # 如果没有 Alpha 通道，创建全不透明
        transformed_image = cv2.warpAffine(
            image, transform_matrix, (output_width, output_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        # 性能优化：简化alpha创建逻辑
        alpha = np.full((output_height, output_width), 255, dtype=np.uint8)

        # 只有在需要时才进行灰度转换
        if len(transformed_image.shape) == 3:
            # 性能优化：使用更高效的方法检查黑色区域
            # 计算所有通道的和，避免灰度转换
            pixel_sum = transformed_image.sum(axis=2)
            alpha = np.where(pixel_sum > 30, 255, 0).astype(np.uint8)

        return transformed_image, alpha

