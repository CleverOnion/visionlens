"""
轻量级人脸检测器
专门为性能优化设计的简化版本
"""

import cv2
import numpy as np
from typing import Optional, Tuple
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("警告: MediaPipe 未安装或版本不兼容，使用备用检测方案")
    MEDIAPIPE_AVAILABLE = False


class FaceDetectorLite:
    """轻量级人脸检测器，专为性能优化"""

    def __init__(self,
                 min_detection_confidence: float = 0.1,
                 min_tracking_confidence: float = 0.1):
        """
        初始化轻量级人脸检测器

        Args:
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.device = "CPU"
        self.last_landmarks = None

        if MEDIAPIPE_AVAILABLE:
            try:
                # 尝试使用 FaceMesh
                if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
                    self.mp_face_mesh = mp.solutions.face_mesh
                    self.face_mesh = self.mp_face_mesh.FaceMesh(
                        max_num_faces=1,
                        refine_landmarks=False,
                        min_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=min_tracking_confidence
                    )
                    self.detector_type = "facemesh"
                else:
                    # 回退到 OpenCV Haar 级联检测器
                    print("MediaPipe 版本不兼容，回退到 OpenCV 检测器")
                    self.init_opencv_detector()
                    self.detector_type = "opencv"
            except Exception as e:
                print(f"MediaPipe 初始化失败: {e}，回退到 OpenCV 检测器")
                self.init_opencv_detector()
                self.detector_type = "opencv"
        else:
            self.init_opencv_detector()
            self.detector_type = "opencv"

    def init_opencv_detector(self):
        """初始化 OpenCV 人脸检测器"""
        try:
            # 使用 OpenCV 的 Haar 级联检测器
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
            self.detector_type = "opencv"
            print("使用 OpenCV Haar 级联检测器（更轻量但精度较低）")
        except Exception as e:
            print(f"OpenCV 检测器初始化失败: {e}")
            self.detector_type = "none"

    def get_device(self) -> str:
        """获取当前使用的设备"""
        return self.device

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        检测人脸并提取关键点（轻量版本）

        Args:
            image: BGR 格式的图像

        Returns:
            关键点数组，形状为 (468, 3)，如果未检测到人脸返回 None
        """
        if self.detector_type == "facemesh":
            return self.detect_with_facemesh(image)
        elif self.detector_type == "opencv":
            return self.detect_with_opencv(image)
        else:
            return None

    def detect_with_facemesh(self, image: np.ndarray) -> Optional[np.ndarray]:
        """使用 FaceMesh 检测"""
        try:
            # MediaPipe 需要 RGB 格式
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 检测
            results = self.face_mesh.process(rgb_image)

            if not results.multi_face_landmarks:
                return None

            # 获取第一个检测到的人脸的关键点
            face_landmarks = results.multi_face_landmarks[0]

            # 获取图像尺寸
            h, w = image.shape[:2]

            # 转换为 numpy 数组
            landmarks = []
            for landmark in face_landmarks.landmark:
                x = landmark.x * w
                y = landmark.y * h
                z = landmark.z * w  # z 坐标通常以宽度为单位
                landmarks.append([x, y, z])

            landmarks_array = np.array(landmarks)
            self.last_landmarks = landmarks_array
            return landmarks_array
        except Exception as e:
            print(f"FaceMesh 检测失败: {e}")
            return None

    def detect_with_opencv(self, image: np.ndarray) -> Optional[np.ndarray]:
        """使用 OpenCV Haar 级联检测器"""
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 检测人脸
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)
            )

            if len(faces) == 0:
                return None

            # 取第一个检测到的人脸
            (x, y, w, h) = faces[0]

            # 生成简化的468个关键点（基于人脸框的估计）
            # 这是一个简化的关键点生成，不精确但足以用于基本的饰品定位
            landmarks = []

            # 生成网格化的关键点
            for i in range(26):  # 26x18 = 468 个点
                for j in range(18):
                    # 在人脸框内均匀分布点
                    px = x + (i / 25) * w
                    py = y + (j / 17) * h
                    pz = 0  # z 坐标设为0
                    landmarks.append([px, py, pz])

            landmarks_array = np.array(landmarks)

            # 确保输出是正确的形状
            if landmarks_array.shape != (468, 3):
                # 如果生成的点数不对，重新生成
                landmarks_array = np.zeros((468, 3))
                for i in range(468):
                    # 在人脸框内随机分布但相对固定
                    px = x + (i % 26) * (w / 25)
                    py = y + (i // 26) * (h / 17)
                    landmarks_array[i] = [px, py, 0]

            self.last_landmarks = landmarks_array
            return landmarks_array
        except Exception as e:
            print(f"OpenCV 检测失败: {e}")
            return None

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
            if self.detector_type == "facemesh":
                # MediaPipe FaceMesh 的精确关键点索引
                return {
                    'left_eye_top': landmarks[159],      # 左眼上方
                    'right_eye_top': landmarks[386],     # 右眼上方
                    'nose_bridge': landmarks[6],         # 鼻梁
                    'chin': landmarks[175],              # 下巴
                    'left_eye_left': landmarks[33],      # 左眼左侧
                    'left_eye_right': landmarks[133],    # 左眼右侧
                    'right_eye_left': landmarks[362],    # 右眼左侧
                    'right_eye_right': landmarks[263],   # 右眼右侧
                    'forehead': landmarks[10],           # 额头
                }
            else:
                # OpenCV 的估算关键点（基于网格分布）
                h, w = 0, 0
                if len(landmarks) > 0:
                    # 找到边界
                    x_coords = landmarks[:, 0]
                    y_coords = landmarks[:, 1]
                    x_min, x_max = np.min(x_coords), np.max(x_coords)
                    y_min, y_max = np.min(y_coords), np.max(y_coords)
                    w = x_max - x_min
                    h = y_max - y_min

                # 基于网格估算关键点位置
                center_x = np.mean(landmarks[:, 0])
                center_y = np.mean(landmarks[:, 1])

                return {
                    'left_eye_top': np.array([center_x - w*0.15, center_y - h*0.05, 0]),
                    'right_eye_top': np.array([center_x + w*0.15, center_y - h*0.05, 0]),
                    'nose_bridge': np.array([center_x, center_y, 0]),
                    'chin': np.array([center_x, center_y + h*0.3, 0]),
                    'left_eye_left': np.array([center_x - w*0.25, center_y - h*0.05, 0]),
                    'left_eye_right': np.array([center_x - w*0.05, center_y - h*0.05, 0]),
                    'right_eye_left': np.array([center_x + w*0.05, center_y - h*0.05, 0]),
                    'right_eye_right': np.array([center_x + w*0.25, center_y - h*0.05, 0]),
                    'forehead': np.array([center_x, center_y - h*0.3, 0]),
                }
        except Exception as e:
            print(f"关键点提取失败: {e}")
            return {}

    def calculate_head_pose(self, landmarks: np.ndarray) -> Tuple[float, float, float]:
        """
        计算头部姿态角（简化版本）

        Args:
            landmarks: 关键点数组

        Returns:
            (pitch, yaw, roll) 角度（度）
        """
        if landmarks is None or len(landmarks) < 468:
            return 0.0, 0.0, 0.0

        try:
            # 获取关键点
            key_points = self.get_key_points(landmarks)
            if not key_points:
                return 0.0, 0.0, 0.0

            left_eye = (key_points['left_eye_left'] + key_points['left_eye_right']) / 2
            right_eye = (key_points['right_eye_left'] + key_points['right_eye_right']) / 2
            nose = key_points['nose_bridge']

            # 计算向量
            eye_vector = right_eye - left_eye
            nose_vector = nose - (left_eye + right_eye) / 2

            # 简化的角度计算（比完整版本更高效）
            yaw = np.degrees(np.arctan2(eye_vector[0], eye_vector[1]))
            pitch = np.degrees(np.arctan2(-nose_vector[1], np.sqrt(nose_vector[0]**2 + nose_vector[2]**2)))
            roll = np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))

            return pitch, yaw, roll
        except:
            return 0.0, 0.0, 0.0