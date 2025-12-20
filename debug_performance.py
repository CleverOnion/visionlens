"""
性能调试脚本
用于分析FPS瓶颈的具体原因
"""

import cv2
import time
import numpy as np
from visionlens.face_detector import FaceDetector
from visionlens.accessory_manager import AccessoryManager

def detailed_performance_debug():
    """详细的性能调试"""
    print("=== VisionLens AR 详细性能调试 ===")

    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误: 无法打开摄像头")
        return

    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"摄像头实际分辨率: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"摄像头FPS: {cap.get(cv2.CAP_PROP_FPS)}")

    # 初始化组件
    face_detector = FaceDetector(
        max_num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )
    accessory_manager = AccessoryManager(assets_dir="./assets")

    # 性能计时器
    frame_times = []
    capture_times = []
    flip_times = []
    detection_times = []
    render_times = []
    display_times = []

    frame_index = 0
    detect_every_n_frames = 5  # 更激进的检测频率
    last_landmarks = None

    print(f"\n开始详细性能分析，检测间隔: {detect_every_n_frames} 帧")
    print("按 'q' 结束分析")
    print("-" * 60)

    try:
        while True:
            total_start = time.time()

            # 1. 摄像头捕获
            capture_start = time.time()
            ret, frame = cap.read()
            capture_time = time.time() - capture_start
            capture_times.append(capture_time)

            if not ret:
                print("无法读取摄像头帧")
                break

            # 2. 图像翻转
            flip_start = time.time()
            frame = cv2.flip(frame, 1)
            flip_time = time.time() - flip_start
            flip_times.append(flip_time)

            # 3. 人脸检测
            landmarks = None
            frame_index += 1

            if frame_index % detect_every_n_frames == 0:
                detection_start = time.time()
                landmarks = face_detector.detect(frame)
                detection_time = time.time() - detection_start
                detection_times.append(detection_time)

                if landmarks is not None:
                    last_landmarks = landmarks
                else:
                    last_landmarks = None
            else:
                landmarks = last_landmarks
                detection_times.append(0)  # 未检测帧

            # 4. 渲染饰品
            render_start = time.time()
            if landmarks is not None:
                frame = accessory_manager.render_accessories(frame, landmarks, face_detector)
            render_time = time.time() - render_start
            render_times.append(render_time)

            # 5. 显示图像
            display_start = time.time()
            cv2.imshow("Debug Performance", frame)
            display_time = time.time() - display_start
            display_times.append(display_time)

            # 总帧时间
            total_time = time.time() - total_start
            frame_times.append(total_time)

            # 每30帧打印一次统计
            if len(frame_times) % 30 == 0:
                print(f"\n--- 第 {len(frame_times)} 帧统计 ---")
                print(f"当前FPS: {1/np.mean(frame_times[-30:]):.1f}")
                print(f"平均捕获时间: {np.mean(capture_times[-30:])*1000:.1f} ms")
                print(f"平均翻转时间: {np.mean(flip_times[-30:])*1000:.1f} ms")
                print(f"平均检测时间: {np.mean(detection_times[-30:])*1000:.1f} ms (非零: {sum(1 for t in detection_times[-30:] if t > 0)})")
                print(f"平均渲染时间: {np.mean(render_times[-30:])*1000:.1f} ms")
                print(f"平均显示时间: {np.mean(display_times[-30:])*1000:.1f} ms")

                # 分析瓶颈
                avg_capture = np.mean(capture_times[-30:])
                avg_flip = np.mean(flip_times[-30:])
                avg_detection = np.mean([t for t in detection_times[-30:] if t > 0])
                avg_render = np.mean(render_times[-30:])
                avg_display = np.mean(display_times[-30:])

                print(f"\n瓶颈分析 (时间占比):")
                total_avg = avg_capture + avg_flip + (avg_detection/detect_every_n_frames) + avg_render + avg_display
                print(f"摄像头捕获: {avg_capture/total_avg*100:.1f}%")
                print(f"图像翻转: {avg_flip/total_avg*100:.1f}%")
                print(f"人脸检测: {(avg_detection/detect_every_n_frames)/total_avg*100:.1f}%")
                print(f"饰品渲染: {avg_render/total_avg*100:.1f}%")
                print(f"图像显示: {avg_display/total_avg*100:.1f}%")

            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n调试被用户中断")
    finally:
        cap.release()
        cv2.destroyAllWindows()

        # 最终统计
        print(f"\n=== 最终性能统计 ===")
        print(f"总帧数: {len(frame_times)}")
        print(f"平均FPS: {1/np.mean(frame_times):.1f}")

        # 详细分析
        print(f"\n各步骤平均时间:")
        print(f"摄像头捕获: {np.mean(capture_times)*1000:.1f} ms")
        print(f"图像翻转: {np.mean(flip_times)*1000:.1f} ms")
        print(f"人脸检测: {np.mean(detection_times)*1000:.1f} ms (实际检测: {np.mean([t for t in detection_times if t > 0])*1000:.1f} ms)")
        print(f"饰品渲染: {np.mean(render_times)*1000:.1f} ms")
        print(f"图像显示: {np.mean(display_times)*1000:.1f} ms")

        print(f"\n优化建议:")
        if np.mean(capture_times) > 0.03:
            print("- 摄像头捕获时间过长，考虑降低分辨率")
        if np.mean(flip_times) > 0.005:
            print("- 图像翻转时间较长，可以考虑移除或优化")
        if np.mean([t for t in detection_times if t > 0]) > 0.05:
            print("- 人脸检测时间过长，增加检测间隔或降低精度")
        if np.mean(render_times) > 0.02:
            print("- 渲染时间过长，优化图像处理算法")

if __name__ == "__main__":
    detailed_performance_debug()