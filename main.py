"""
VisionLens AR - 主程序入口
基于 MediaPipe 的实时虚拟试戴系统
"""

import cv2
import time
import sys
from visionlens.face_detector import FaceDetector
from visionlens.accessory_manager import AccessoryManager


def main():
    """主程序入口"""
    print("VisionLens AR - 基于 MediaPipe 的实时虚拟试戴系统")
    print("=" * 50)

    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误: 无法打开摄像头")
        sys.exit(1)

    # 设置摄像头分辨率（可选，提高性能）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 初始化人脸检测器（优先使用 GPU）
    face_detector = FaceDetector(
        max_num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 初始化饰品管理器
    print("正在加载饰品资源...")
    accessory_manager = AccessoryManager(assets_dir="./assets")

    glasses_count = len(accessory_manager.glasses_list)
    beard_count = len(accessory_manager.beard_list)

    print(f"已加载 {glasses_count} 个眼镜，{beard_count} 个胡子")

    if glasses_count == 0 and beard_count == 0:
        print("警告: 未找到任何饰品资源文件")
        print("请将 PNG 格式的眼镜和胡子文件放入 ./assets/glasses/ 和 ./assets/beard/ 目录")

    # 创建窗口
    window_name = "VisionLens AR - 按 'g' 切换眼镜, 'b' 切换胡子, 'q' 退出"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # FPS 计算
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0

    # 人脸检测节流：每 N 帧检测一次，其余复用上次关键点
    detect_every_n_frames = 1  # 可以根据机器性能调整，例如 2~5
    frame_index = 0
    last_landmarks = None

    print("\n程序已启动！")
    print("按键说明:")
    print("  'g' - 切换眼镜")
    print("  'b' - 切换胡子")
    print("  'q' - 退出程序")
    print("-" * 50)

    try:
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法读取摄像头帧")
                break

            # 水平翻转（镜像效果）
            frame = cv2.flip(frame, 1)

            # 检测人脸（节流：每 detect_every_n_frames 帧检测一次）
            frame_index += 1
            landmarks = None
            if frame_index % detect_every_n_frames == 0:
                landmarks = face_detector.detect(frame)
                # 如果当前帧成功检测到人脸，则更新缓存
                if landmarks is not None:
                    last_landmarks = landmarks
                else:
                    # 本帧未检测到人脸，清空缓存，避免误用旧结果
                    last_landmarks = None
            else:
                # 非检测帧：复用上一次成功检测到的人脸关键点
                landmarks = last_landmarks

            # 渲染饰品
            if landmarks is not None:
                frame = accessory_manager.render_accessories(
                    frame, landmarks, face_detector
                )

            # 计算 FPS
            fps_frame_count += 1
            if fps_frame_count >= 30:
                fps_end_time = time.time()
                fps = fps_frame_count / (fps_end_time - fps_start_time)
                fps_frame_count = 0
                fps_start_time = fps_end_time

            # 在帧上绘制信息
            device_text = face_detector.get_device()
            detect_info = f"/{detect_every_n_frames}f" if detect_every_n_frames > 1 else "/1f"
            info_text = f"FPS: {fps:.1f} | Device: {device_text} | Detect: 1{detect_info}"
            status_text = accessory_manager.get_status()

            # 根据设备类型选择颜色（GPU 用绿色，CPU 用黄色）
            device_color = (0, 255, 0) if device_text == "GPU" else (0, 255, 255)

            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, device_color, 2)
            cv2.putText(frame, status_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 显示帧
            cv2.imshow(window_name, frame)

            # 处理按键
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
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        cap.release()
        cv2.destroyAllWindows()
        print("资源已释放，程序退出")


if __name__ == "__main__":
    main()