"""
饰品管理器模块
负责加载、管理和渲染饰品
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from .face_detector import FaceDetector
from .transform_utils import (
    calculate_glasses_transform,
    calculate_beard_transform,
    apply_transform
)
from .effects import blend_multiple_accessories


class AccessoryManager:
    """饰品管理器，负责加载和管理眼镜、胡子等饰品"""
    
    def __init__(self, assets_dir: str = "./assets"):
        """
        初始化饰品管理器

        Args:
            assets_dir: 资源文件目录
        """
        self.assets_dir = Path(assets_dir)
        self.glasses_dir = self.assets_dir / "glasses"
        self.beard_dir = self.assets_dir / "beard"

        # 加载所有可用的饰品
        self.glasses_list = self._load_accessories(self.glasses_dir)
        self.beard_list = self._load_accessories(self.beard_dir)

        # 当前激活的饰品索引
        # 如果有可用的饰品，默认激活第一个（索引0），否则为-1表示未激活
        self.current_glasses_idx = 0 if len(self.glasses_list) > 0 else -1
        self.current_beard_idx = 0 if len(self.beard_list) > 0 else -1

        # 缓存的饰品图像
        self.glasses_cache: Dict[str, np.ndarray] = {}
        self.beard_cache: Dict[str, np.ndarray] = {}

        # 性能优化：缓存变换矩阵，避免重复计算
        self.last_glasses_transform = None
        self.last_beard_transform = None
        self.last_landmarks_hash = None
        
    def _load_accessories(self, directory: Path) -> List[str]:
        """
        加载指定目录下的所有 PNG 文件
        
        Args:
            directory: 目录路径
            
        Returns:
            PNG 文件路径列表
        """
        if not directory.exists():
            return []
            
        png_files = list(directory.glob("*.png"))
        return sorted([str(f) for f in png_files])
    
    def _load_image(self, file_path: str) -> Optional[np.ndarray]:
        """
        加载图像文件（支持 Alpha 通道，优化缓存性能）

        Args:
            file_path: 图像文件路径

        Returns:
            BGRA 格式的图像，如果加载失败返回 None
        """
        # 性能优化：检查缓存，避免重复加载和copy()操作
        cache = self.glasses_cache if "glasses" in str(file_path).lower() else self.beard_cache
        if file_path in cache:
            return cache[file_path]  # 直接返回引用，避免copy()

        # 检查文件是否存在（使用 Path 对象处理路径）
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            print(f"警告: 图片文件不存在: {file_path}")
            return None

        # 使用 cv2.IMREAD_UNCHANGED 保留 Alpha 通道
        image = cv2.imread(str(file_path_obj), cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"警告: 无法加载图片文件: {file_path}")
            return None

        # 确保图像有正确的维度
        if len(image.shape) < 2:
            print(f"警告: 图片格式不正确: {file_path}")
            return None

        # 性能优化：简化图像格式处理逻辑
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                # 已经是 BGRA 格式，直接使用
                pass
            elif image.shape[2] == 3:
                # 如果是 BGR 图像，添加全不透明的 Alpha 通道
                alpha = np.full((image.shape[0], image.shape[1]), 255, dtype=np.uint8)
                image = np.dstack([image, alpha])
            else:
                print(f"警告: 不支持的图片通道数: {image.shape[2]}")
                return None
        else:
            # 灰度图，转换为 BGR 并添加 Alpha
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            alpha = np.full((image.shape[0], image.shape[1]), 255, dtype=np.uint8)
            image = np.dstack([image, alpha])

        # 性能优化：直接缓存到正确的字典
        cache[file_path] = image

        return image
    
    def toggle_glasses(self):
        """切换眼镜"""
        if len(self.glasses_list) == 0:
            return

        self.current_glasses_idx += 1
        if self.current_glasses_idx >= len(self.glasses_list):
            self.current_glasses_idx = -1  # 关闭眼镜

        # 性能优化：切换饰品时清空变换矩阵缓存
        self.last_glasses_transform = None

    def toggle_beard(self):
        """切换胡子"""
        if len(self.beard_list) == 0:
            return

        self.current_beard_idx += 1
        if self.current_beard_idx >= len(self.beard_list):
            self.current_beard_idx = -1  # 关闭胡子

        # 性能优化：切换饰品时清空变换矩阵缓存
        self.last_beard_transform = None
    
    def render_accessories(self,
                          frame: np.ndarray,
                          landmarks: np.ndarray,
                          face_detector: FaceDetector) -> np.ndarray:
        """
        渲染所有激活的饰品到帧上（性能优化版本）

        Args:
            frame: 输入帧（BGR）
            landmarks: 人脸关键点
            face_detector: FaceDetector 实例

        Returns:
            渲染后的帧（BGR）
        """
        if landmarks is None or len(landmarks) < 468:
            return frame

        h, w = frame.shape[:2]
        accessories = []

        # 性能优化：计算landmarks哈希值，用于缓存判断
        landmarks_hash = hash(landmarks.tobytes())
        landmarks_changed = landmarks_hash != self.last_landmarks_hash
        self.last_landmarks_hash = landmarks_hash

        # 渲染眼镜
        if self.current_glasses_idx >= 0 and self.current_glasses_idx < len(self.glasses_list):
            glasses_path = self.glasses_list[self.current_glasses_idx]
            glasses_image = self._load_image(glasses_path)

            if glasses_image is not None:
                glasses_h, glasses_w = glasses_image.shape[:2]

                # 性能优化：缓存变换矩阵，只有关键点变化时才重新计算
                if landmarks_changed or self.last_glasses_transform is None:
                    transform_matrix = calculate_glasses_transform(
                        landmarks, face_detector, glasses_w, glasses_h
                    )
                    self.last_glasses_transform = transform_matrix
                else:
                    transform_matrix = self.last_glasses_transform

                if transform_matrix is not None:
                    transformed_image, transformed_alpha = apply_transform(
                        glasses_image, transform_matrix, (h, w)
                    )

                    # 性能优化：简化验证逻辑
                    if (transformed_image is not None and transformed_alpha is not None and
                        transformed_image.shape[:2] == (h, w) and transformed_alpha.shape[:2] == (h, w)):

                        # 提取 BGR 和 Alpha
                        if transformed_image.shape[2] == 4:
                            foreground_bgr = transformed_image[:, :, :3]
                            # 确保 alpha 通道有有效值
                            if transformed_alpha.max() > 0:
                                accessories.append((foreground_bgr, transformed_alpha))

        # 渲染胡子
        if self.current_beard_idx >= 0 and self.current_beard_idx < len(self.beard_list):
            beard_path = self.beard_list[self.current_beard_idx]
            beard_image = self._load_image(beard_path)

            if beard_image is not None:
                beard_h, beard_w = beard_image.shape[:2]

                # 性能优化：缓存变换矩阵，只有关键点变化时才重新计算
                if landmarks_changed or self.last_beard_transform is None:
                    transform_matrix = calculate_beard_transform(
                        landmarks, face_detector, beard_w, beard_h
                    )
                    self.last_beard_transform = transform_matrix
                else:
                    transform_matrix = self.last_beard_transform

                if transform_matrix is not None:
                    transformed_image, transformed_alpha = apply_transform(
                        beard_image, transform_matrix, (h, w)
                    )

                    # 性能优化：简化验证逻辑
                    if (transformed_image is not None and transformed_alpha is not None and
                        transformed_image.shape[:2] == (h, w) and transformed_alpha.shape[:2] == (h, w)):

                        # 提取 BGR 和 Alpha
                        if transformed_image.shape[2] == 4:
                            foreground_bgr = transformed_image[:, :, :3]
                            # 确保 alpha 通道有有效值
                            if transformed_alpha.max() > 0:
                                accessories.append((foreground_bgr, transformed_alpha))

        # 融合所有饰品
        if accessories:
            frame = blend_multiple_accessories(frame, accessories)

        return frame
    
    def get_status(self) -> str:
        """
        获取当前饰品状态字符串
        
        Returns:
            状态描述字符串
        """
        glasses_status = "None"
        if self.current_glasses_idx >= 0 and self.current_glasses_idx < len(self.glasses_list):
            glasses_name = Path(self.glasses_list[self.current_glasses_idx]).stem
            glasses_status = f"{glasses_name} ({self.current_glasses_idx + 1}/{len(self.glasses_list)})"
        
        beard_status = "None"
        if self.current_beard_idx >= 0 and self.current_beard_idx < len(self.beard_list):
            beard_name = Path(self.beard_list[self.current_beard_idx]).stem
            beard_status = f"{beard_name} ({self.current_beard_idx + 1}/{len(self.beard_list)})"
        
        return f"Glasses: {glasses_status} | Beard: {beard_status}"

