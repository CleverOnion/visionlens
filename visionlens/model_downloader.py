"""
MediaPipe 模型文件下载工具
"""

import os
import urllib.request
from pathlib import Path


def get_model_path(model_name: str = "face_landmarker.task") -> str:
    """
    获取模型文件路径，如果不存在则自动下载
    
    Args:
        model_name: 模型文件名
        
    Returns:
        模型文件的完整路径
    """
    # 模型文件存储目录
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / model_name
    
    # 如果模型文件不存在，则下载
    if not model_path.exists():
        print(f"正在下载模型文件: {model_name}...")
        download_model(model_name, model_path)
    
    return str(model_path)


def download_model(model_name: str, save_path: Path):
    """
    从 MediaPipe 官方资源下载模型文件
    
    Args:
        model_name: 模型文件名
        save_path: 保存路径
    """
    # MediaPipe 官方模型文件 URL（从 Google Cloud Storage）
    model_urls = {
        "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "face_landmarker_v2.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker_v2/float16/1/face_landmarker_v2.task",
    }
    
    model_url = model_urls.get(model_name, model_urls["face_landmarker.task"])
    
    try:
        print(f"正在从 MediaPipe 官方资源下载模型: {model_name}...")
        print(f"URL: {model_url}")
        
        # 显示下载进度
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            print(f"\r下载进度: {percent:.1f}%", end='', flush=True)
        
        urllib.request.urlretrieve(model_url, save_path, show_progress)
        print(f"\n模型文件已成功保存到: {save_path}")
    except Exception as e:
        print(f"\n下载模型失败: {e}")
        print("\n请手动下载模型文件:")
        print(f"1. 访问: https://developers.google.com/mediapipe/solutions/vision/face_landmarker")
        print(f"2. 或直接下载: {model_url}")
        print(f"3. 将文件保存到: {save_path}")
        print("\n如果下载失败，您也可以:")
        print("   - 检查网络连接")
        print("   - 使用代理或 VPN")
        print("   - 手动下载后放置到指定目录")
        raise

