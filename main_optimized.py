"""
VisionLens AR - 极限性能优化版本
基于 MediaPipe 的实时虚拟试戴系统
采用更激进的优化策略以达到更高FPS
"""

import cv2
import time
import sys
import numpy as np
from visionlens.face_detector import FaceDetector
from visionlens.accessory_manager import AccessoryManager


def main():
    """极限性能优化主程序"""
    print("VisionLens AR - 极限性能优化版本")
    print("=" * 50)

    # 初始化摄像头（更激进的设置）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误: 无法打开摄像头")
        sys.exit(1)

    # 极限优化：更低的分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # 降到320x240
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 60)  # 目标60FPS

    # 获取实际设置
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头分辨率: {actual_width}x{actual_height}")

    # 极限优化：更低的人脸检测精度
    face_detector = FaceDetector(
        max_num_faces=1,
        min_face_detection_confidence=0.1,  # 极低置信度
        min_face_presence_confidence=0.1,   # 极低置信度
        min_tracking_confidence=0.1         # 极低置信度
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
    window_name = "VisionLens AR - 极限优化版"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # FPS 计算
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0

    # 极限优化：更激进的检测频率
    detect_every_n_frames = 8  # 每8帧检测一次
    frame_index = 0
    last_landmarks = None
    last_landmarks_age = 0

    # 性能监控
    performance_samples = []
    target_fps = 30
    adjust_threshold = 15  # 每15帧调整一次

    print(f"\n程序已启动！（极限优化模式）")
    print("优化策略:")
    print(f"  - 分辨率: {actual_width}x{actual_height}")
    print(f"  - 检测频率: 每{detect_every_n_frames}帧检测一次")
    print(f"  - 检测精度: 极低阈值")
    print("按键说明:")
    print("  'g' - 切换眼镜")
    print("  'b' - 切换胡子")
    print("  '+' - 增加检测频率")
    print("  '-' - 减少检测频率")
    print("  'q' - 退出程序")
    print("-" * 50)

    # 预创建一些变量以提高性能
    frame_times = []

    try:
        while True:
            loop_start = time.time()

            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法读取摄像头帧")
                break

            # 极限优化：移除翻转操作（如果不需要镜像效果）
            # frame = cv2.flip(frame, 1)

            # 极限优化：更激进的检测策略
            frame_index += 1
            should_detect = (frame_index % detect_every_n_frames == 0)

            # 更严格的缓存过期：超过20帧重新检测
            if last_landmarks is not None and last_landmarks_age > 20:
                should_detect = True
                last_landmarks_age = 0

            landmarks = None
            if should_detect:
                landmarks = face_detector.detect(frame)
                if landmarks is not None:
                    last_landmarks = landmarks
                    last_landmarks_age = 0
                else:
                    # 检测失败时增加缓存年龄但不清空
                    if last_landmarks is not None:
                        last_landmarks_age += 1
            else:
                # 非检测帧：复用缓存
                if last_landmarks is not None:
                    landmarks = last_landmarks
                    last_landmarks_age += 1

            # 渲染饰品
            if landmarks is not None:
                frame = accessory_manager.render_accessories(
                    frame, landmarks, face_detector
                )

            # 计算 FPS
            fps_frame_count += 1
            if fps_frame_count >= 30:
                fps_end_time = time.time()
                current_fps = fps_frame_count / (fps_end_time - fps_start_time)
                fps = current_fps
                fps_frame_count = 0
                fps_start_time = fps_end_time

                # 自适应性能调节（更激进）
                performance_samples.append(current_fps)
                if len(performance_samples) >= adjust_threshold:
                    avg_fps = sum(performance_samples) / len(performance_samples)

                    # 更激进的调整策略
                    if avg_fps < 20:  # FPS太低
                        if detect_every_n_frames < 12:  # 最大12帧
                            detect_every_n_frames += 2  # 每次增加2帧
                            print(f"性能优化：大幅降低检测频率，每{detect_every_n_frames}帧检测一次")
                    elif avg_fps > 35:  # FPS很充足
                        if detect_every_n_frames > 4:  # 最小4帧
                            detect_every_n_frames -= 1
                            print(f"性能优化：提高检测频率，每{detect_every_n_frames}帧检测一次")

                    performance_samples.clear()

            # 计算实时帧时间
            loop_time = time.time() - loop_start
            frame_times.append(loop_time)
            if len(frame_times) > 30:
                frame_times.pop(0)

            # 在帧上绘制信息
            device_text = face_detector.get_device()
            detect_info = f"/{detect_every_n_frames}f"

            # 极限优化：简化文本绘制
            info_text = f"FPS: {fps:.1f} | {device_text} | Detect: 1{detect_info}"
            status_text = accessory_manager.get_status()

            # 根据设备类型选择颜色
            device_color = (0, 255, 0) if device_text == "GPU" else (0, 255, 255)

            # 性能警告
            if fps < 15:
                warning_color = (0, 0, 255)  # 红色警告
                info_color = warning_color
            elif fps < 25:
                warning_color = (0, 165, 255)  # 橙色警告
                info_color = warning_color
            else:
                info_color = device_color

            cv2.putText(frame, info_text, (5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, info_color, 1)
            cv2.putText(frame, status_text, (5, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # 显示平均帧时间
            if frame_times:
                avg_frame_time = sum(frame_times) / len(frame_times)
                frame_time_text = f"Frame: {avg_frame_time*1000:.1f}ms"
                cv2.putText(frame, frame_time_text, (5, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

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
            elif key == ord('+'):
                if detect_every_n_frames > 2:
                    detect_every_n_frames -= 1
                    print(f"手动增加检测频率: 每{detect_every_n_frames}帧检测一次")
            elif key == ord('-'):
                if detect_every_n_frames < 15:
                    detect_every_n_frames += 1
                    print(f"手动减少检测频率: 每{detect_every_n_frames}帧检测一次")

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

        # 最终统计
        print(f"\n=== 极限优化版性能统计 ===")
        print(f"最终检测频率: 每{detect_every_n_frames}帧检测一次")
        if frame_times:
            avg_frame_time = sum(frame_times) / len(frame_times)
            print(f"平均帧处理时间: {avg_frame_time*1000:.1f} ms")
            print(f"理论最大FPS: {1/avg_frame_time:.1f}")


if __name__ == "__main__":
    main()