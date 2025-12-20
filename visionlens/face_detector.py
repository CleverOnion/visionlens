"""
人脸检测与关键点提取模块
使用 MediaPipe Face Landmarker 提取 468 个 3D 关键点
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision.core import image as mp_image
from .model_downloader import get_model_path


def check_gpu_available() -> bool:
    """
    检查 GPU 是否可用
    
    Returns:
        如果 GPU 可用返回 True，否则返回 False
    """
    try:
        import subprocess
        import platform
        
        # 在 Windows 上检查 NVIDIA GPU
        if platform.system() == "Windows":
            try:
                result = subprocess.run(
                    ["nvidia-smi"], 
                    capture_output=True, 
                    text=True, 
                    timeout=2
                )
                if result.returncode == 0 and "NVIDIA" in result.stdout:
                    return True
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
                pass
        
        # 在 Linux 上检查 NVIDIA GPU
        if platform.system() == "Linux":
            try:
                result = subprocess.run(
                    ["nvidia-smi"], 
                    capture_output=True, 
                    text=True, 
                    timeout=2
                )
                if result.returncode == 0:
                    return True
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
                pass
        
        return False
    except Exception:
        return False


def get_delegate() -> base_options.BaseOptions.Delegate:
    """
    获取 MediaPipe 委托（优先使用 GPU）
    
    Returns:
        优先返回 GPU delegate，如果 GPU 不可用 MediaPipe 会自动回退到 CPU
    """
    # MediaPipe 会自动检测 GPU 是否可用
    # 如果 GPU 不可用，MediaPipe 会自动回退到 CPU，不会抛出异常
    # 所以我们直接返回 GPU delegate，让 MediaPipe 自动处理
    return base_options.BaseOptions.Delegate.GPU


class FaceDetector:
    """MediaPipe 人脸检测器，提取 468 个 3D 关键点"""
    
    # 关键点索引映射
    LEFT_EYE_TOP = 159
    RIGHT_EYE_TOP = 386
    NOSE_BRIDGE = 6
    CHIN = 175
    LEFT_EYE_LEFT = 33
    LEFT_EYE_RIGHT = 133
    RIGHT_EYE_LEFT = 362
    RIGHT_EYE_RIGHT = 263
    FOREHEAD = 10
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 max_num_faces: int = 1,
                 min_face_detection_confidence: float = 0.5,
                 min_face_presence_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 output_face_blendshapes: bool = False,
                 output_facial_transformation_matrixes: bool = False):
        """
        初始化人脸检测器
        
        Args:
            model_path: 模型文件路径，如果为 None 则自动下载
            max_num_faces: 最大检测人脸数
            min_face_detection_confidence: 最小检测置信度
            min_face_presence_confidence: 最小存在置信度
            min_tracking_confidence: 最小跟踪置信度
            output_face_blendshapes: 是否输出面部混合形状
            output_facial_transformation_matrixes: 是否输出面部变换矩阵
        """
        # 获取模型文件路径
        if model_path is None:
            model_path = get_model_path("face_landmarker.task")
        
        # 优先使用 GPU（如果可用）
        # MediaPipe 会自动检测 GPU 是否可用，如果不可用会自动回退到 CPU
        delegate = get_delegate()
        self.delegate = delegate  # 保存 delegate 信息
        
        # 检测 GPU 是否可用
        gpu_available = check_gpu_available()
        
        # 如果请求使用 GPU 且检测到 GPU 可用，则使用 GPU
        # 注意：MediaPipe 可能会在运行时回退到 CPU，但我们根据初始配置显示
        if delegate == base_options.BaseOptions.Delegate.GPU and gpu_available:
            print(f"正在初始化 MediaPipe（使用 GPU 加速）...")
            self.device = "GPU"
        else:
            print(f"正在初始化 MediaPipe（使用 CPU）...")
            self.device = "CPU"
        
        # 创建 FaceLandmarkerOptions
        base_opts = base_options.BaseOptions(
            model_asset_path=model_path,
            delegate=delegate  # MediaPipe 会自动处理 GPU 不可用的情况
        )
        
        options = vision.FaceLandmarkerOptions(
            base_options=base_opts,
            output_face_blendshapes=output_face_blendshapes,
            output_facial_transformation_matrixes=output_facial_transformation_matrixes,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=vision.RunningMode.VIDEO  # 视频模式
        )
        
        # 创建 FaceLandmarker
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        self.timestamp_ms = 0
    
    def get_device(self) -> str:
        """
        获取当前使用的设备（GPU 或 CPU）
        
        Returns:
            设备名称字符串："GPU" 或 "CPU"
        """
        return getattr(self, 'device', 'CPU')
        
    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        检测人脸并提取关键点
        
        Args:
            image: BGR 格式的图像
            
        Returns:
            关键点数组，形状为 (468, 3)，每个关键点包含 (x, y, z) 坐标
            如果未检测到人脸，返回 None
        """
        # MediaPipe 需要 RGB 格式
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 转换为 MediaPipe Image 格式
        mp_img = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb_image)
        
        # 检测人脸关键点
        self.timestamp_ms += 33  # 假设 30 FPS
        detection_result = self.face_landmarker.detect_for_video(mp_img, self.timestamp_ms)
        
        if not detection_result.face_landmarks:
            return None
            
        # 提取第一个检测到的人脸的关键点
        face_landmarks = detection_result.face_landmarks[0]
        
        # 获取图像尺寸
        h, w = image.shape[:2]
        
        # 转换为 numpy 数组，包含 (x, y, z) 坐标
        landmarks = []
        for landmark in face_landmarks:
            x = landmark.x * w
            y = landmark.y * h
            z = landmark.z * w  # z 坐标通常以宽度为单位
            landmarks.append([x, y, z])
            
        return np.array(landmarks)
    
    def get_key_points(self, landmarks: np.ndarray) -> dict:
        """
        获取关键特征点的坐标
        
        Args:
            landmarks: 468 个关键点的数组
            
        Returns:
            包含关键点坐标的字典
        """
        if landmarks is None or len(landmarks) < 468:
            return {}
        
        try:
            return {
                'left_eye_top': landmarks[self.LEFT_EYE_TOP],
                'right_eye_top': landmarks[self.RIGHT_EYE_TOP],
                'nose_bridge': landmarks[self.NOSE_BRIDGE],
                'chin': landmarks[self.CHIN],
                'left_eye_left': landmarks[self.LEFT_EYE_LEFT],
                'left_eye_right': landmarks[self.LEFT_EYE_RIGHT],
                'right_eye_left': landmarks[self.RIGHT_EYE_LEFT],
                'right_eye_right': landmarks[self.RIGHT_EYE_RIGHT],
                'forehead': landmarks[self.FOREHEAD],
            }
        except (IndexError, KeyError):
            return {}
    
    def calculate_head_pose(self, landmarks: np.ndarray) -> Tuple[float, float, float]:
        """
        计算头部姿态角（Pitch, Yaw, Roll）
        
        Args:
            landmarks: 468 个关键点的数组
            
        Returns:
            (pitch, yaw, roll) 角度（弧度）
        """
        if landmarks is None or len(landmarks) < 468:
            return 0.0, 0.0, 0.0
            
        key_points = self.get_key_points(landmarks)
        
        if not key_points:
            return 0.0, 0.0, 0.0
        
        # 计算 Roll（绕 Z 轴旋转）- 基于两眼连线
        left_eye_center = (key_points['left_eye_left'] + key_points['left_eye_right']) / 2
        right_eye_center = (key_points['right_eye_left'] + key_points['right_eye_right']) / 2
        eye_vector = right_eye_center - left_eye_center
        roll = np.arctan2(eye_vector[1], eye_vector[0])
        
        # 计算 Pitch（绕 X 轴旋转）- 基于鼻梁到下巴的向量
        nose_to_chin = key_points['chin'] - key_points['nose_bridge']
        pitch = np.arctan2(nose_to_chin[2], nose_to_chin[1])
        
        # 计算 Yaw（绕 Y 轴旋转）- 基于两眼中心到鼻梁的向量
        eye_center = (left_eye_center + right_eye_center) / 2
        eye_to_nose = key_points['nose_bridge'] - eye_center
        yaw = np.arctan2(eye_to_nose[2], eye_to_nose[0])
        
        return pitch, yaw, roll
    
    def calculate_face_dimensions(self, landmarks: np.ndarray) -> dict:
        """
        计算人脸尺寸（用于缩放计算）
        
        Args:
            landmarks: 468 个关键点的数组
            
        Returns:
            包含人脸尺寸信息的字典
        """
        if landmarks is None or len(landmarks) < 468:
            return {}
            
        key_points = self.get_key_points(landmarks)
        
        if not key_points:
            return {}
        
        # 计算两眼间距离
        left_eye_center = (key_points['left_eye_left'] + key_points['left_eye_right']) / 2
        right_eye_center = (key_points['right_eye_left'] + key_points['right_eye_right']) / 2
        eye_distance = np.linalg.norm(right_eye_center - left_eye_center)
        
        # 计算鼻梁到下巴的距离
        nose_to_chin_distance = np.linalg.norm(
            key_points['chin'] - key_points['nose_bridge']
        )
        
        # 计算额头到下巴的距离（人脸高度）
        face_height = np.linalg.norm(
            key_points['chin'] - key_points['forehead']
        )
        
        return {
            'eye_distance': eye_distance,
            'nose_to_chin_distance': nose_to_chin_distance,
            'face_height': face_height,
        }
