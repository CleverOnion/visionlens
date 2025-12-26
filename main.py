"""
VisionLens AR - 主程序入口
基于 MediaPipe 的实时虚拟试戴系统
"""

import cv2
import time
import numpy as np
from visionlens.face_detector import FaceDetector
from visionlens.accessory_manager import AccessoryManager


def main():
    """主程序入口"""
    print("=" * 50)
    print("VisionLens AR - 实时虚拟试戴系统")
    print("=" * 50)

    # 初始化摄像头
    print("正在初始化摄像头...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("错误: 无法打开摄像头")
        return

    # 设置摄像头分辨率（可根据性能调整）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头分辨率: {actual_width}x{actual_height}")

    # 初始化人脸检测器
    print("正在初始化人脸检测器...")
    face_detector = FaceDetector()
    device = face_detector.get_device()
    print(f"使用设备: {device}")

    # 初始化饰品管理器
    print("正在加载饰品资源...")
    accessory_manager = AccessoryManager(assets_dir="./assets")

    glasses_count = len(accessory_manager.glasses_list)
    beard_count = len(accessory_manager.beard_list)
    print(f"已加载 {glasses_count} 个眼镜样式, {beard_count} 个胡子样式")

    if glasses_count == 0 and beard_count == 0:
        print("警告: 未找到任何饰品资源")
        print(f"请将 PNG 图片放置到 ./assets/glasses/ 和 ./assets/beard/ 目录")

    print("\n控制说明:")
    print("  'g' - 切换眼镜样式")
    print("  'b' - 切换胡子样式")
    print("  'q' - 退出程序")
    print("=" * 50)

    # 性能优化：帧检测节流（每 N 帧检测一次人脸）
    DETECTION_EVERY_N_FRAMES = 1  # 可根据性能调整

    # FPS 计算变量
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0
    last_landmarks = None

    # 帧计数器（用于检测节流）
    frame_count = 0

    print("\n启动摄像头窗口...\n")

    try:
        while True:
            # 捕获帧
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法读取摄像头帧")
                break

            # 水平翻转（镜像效果）
            frame = cv2.flip(frame, 1)

            # 性能优化：按节流间隔检测人脸
            if frame_count % DETECTION_EVERY_N_FRAMES == 0:
                landmarks = face_detector.detect(frame)
                if landmarks is not None:
                    last_landmarks = landmarks
            else:
                landmarks = last_landmarks

            frame_count += 1

            # 渲染饰品
            if landmarks is not None:
                frame = accessory_manager.render_accessories(frame, landmarks, face_detector)

            # 计算 FPS
            fps_frame_count += 1
            fps_elapsed = time.time() - fps_start_time
            if fps_elapsed >= 1.0:
                current_fps = fps_frame_count / fps_elapsed
                fps_frame_count = 0
                fps_start_time = time.time()

            # 在帧上显示信息
            # FPS
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 设备信息
            cv2.putText(frame, f"Device: {device}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 饰品状态
            status = accessory_manager.get_status()
            cv2.putText(frame, status, (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 显示帧
            cv2.imshow("VisionLens AR", frame)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n退出程序...")
                break
            elif key == ord('g'):
                accessory_manager.toggle_glasses()
                print(f"切换眼镜: {accessory_manager.get_status()}")
            elif key == ord('b'):
                accessory_manager.toggle_beard()
                print(f"切换胡子: {accessory_manager.get_status()}")

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
    finally:
        # 清理资源
        cap.release()
        cv2.destroyAllWindows()
        print("资源已释放，程序退出")


if __name__ == "__main__":
    main()
