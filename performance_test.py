"""
性能测试脚本
用于测试和验证FPS优化效果
"""

import cv2
import time
import numpy as np
from visionlens.face_detector import FaceDetector
from visionlens.accessory_manager import AccessoryManager

def test_performance():
    """测试应用程序性能"""
    print("=== VisionLens AR 性能测试 ===")

    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误: 无法打开摄像头")
        return

    # 设置优化后的摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 初始化人脸检测器（优化参数）
    face_detector = FaceDetector(
        max_num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )

    # 初始化饰品管理器
    print("正在加载饰品资源...")
    accessory_manager = AccessoryManager(assets_dir="./assets")

    print(f"已加载 {len(accessory_manager.glasses_list)} 个眼镜，{len(accessory_manager.beard_list)} 个胡子")

    # 如果没有饰品，创建虚拟测试图像
    if len(accessory_manager.glasses_list) == 0:
        print("警告: 未找到饰品，创建虚拟测试图像")
        # 创建一个简单的测试眼镜
        test_glasses = np.zeros((100, 200, 4), dtype=np.uint8)
        test_glasses[:, :, 0] = 100  # B
        test_glasses[:, :, 1] = 100  # G
        test_glasses[:, :, 2] = 255  # R
        test_glasses[:, :, 3] = 200  # Alpha
        # 保存临时文件
        cv2.imwrite("assets/test_glasses.png", test_glasses)
        accessory_manager.glasses_list = ["assets/test_glasses.png"]

    # 性能测试参数
    test_duration = 30  # 测试30秒
    frame_times = []
    detection_times = []
    render_times = []

    # 自适应参数
    detect_every_n_frames = 3
    frame_index = 0
    last_landmarks = None
    last_landmarks_age = 0

    print(f"\n开始性能测试，持续 {test_duration} 秒...")
    print("请确保您的脸在摄像头前")
    print("按 'q' 提前结束测试")
    print("-" * 50)

    start_time = time.time()

    try:
        while time.time() - start_time < test_duration:
            loop_start = time.time()

            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法读取摄像头帧")
                break

            # 水平翻转
            frame = cv2.flip(frame, 1)

            # 自适应检测
            frame_index += 1
            should_detect = (frame_index % detect_every_n_frames == 0)

            if last_landmarks is not None and last_landmarks_age > 15:
                should_detect = True
                last_landmarks_age = 0

            landmarks = None
            if should_detect:
                detect_start = time.time()
                landmarks = face_detector.detect(frame)
                detection_times.append(time.time() - detect_start)

                if landmarks is not None:
                    last_landmarks = landmarks
                    last_landmarks_age = 0
                else:
                    if last_landmarks is not None:
                        last_landmarks_age += 1
            else:
                if last_landmarks is not None:
                    landmarks = last_landmarks
                    last_landmarks_age += 1

            # 渲染饰品
            render_start = time.time()
            if landmarks is not None:
                frame = accessory_manager.render_accessories(
                    frame, landmarks, face_detector
                )
            render_times.append(time.time() - render_start)

            # 显示帧
            cv2.imshow("Performance Test - VisionLens AR", frame)

            # 计算帧时间
            frame_time = time.time() - loop_start
            frame_times.append(frame_time)

            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n测试被用户中断")
    finally:
        cap.release()
        cv2.destroyAllWindows()

        # 计算统计信息
        total_time = time.time() - start_time
        total_frames = len(frame_times)

        if total_frames > 0:
            avg_fps = total_frames / total_time
            avg_frame_time = sum(frame_times) / len(frame_times) * 1000  # 转换为毫秒
            max_frame_time = max(frame_times) * 1000
            min_frame_time = min(frame_times) * 1000

            print(f"\n=== 性能测试结果 ===")
            print(f"测试时长: {total_time:.1f} 秒")
            print(f"总帧数: {total_frames}")
            print(f"平均 FPS: {avg_fps:.1f}")
            print(f"平均帧处理时间: {avg_frame_time:.1f} ms")
            print(f"最大帧处理时间: {max_frame_time:.1f} ms")
            print(f"最小帧处理时间: {min_frame_time:.1f} ms")

            if detection_times:
                avg_detection_time = sum(detection_times) / len(detection_times) * 1000
                print(f"平均人脸检测时间: {avg_detection_time:.1f} ms")
                print(f"检测频率: 每 {detect_every_n_frames} 帧检测一次")

            if render_times:
                avg_render_time = sum(render_times) / len(render_times) * 1000
                print(f"平均渲染时间: {avg_render_time:.1f} ms")

            # 性能评级
            if avg_fps >= 25:
                print("性能评级: 优秀 (≥25 FPS)")
            elif avg_fps >= 20:
                print("性能评级: 良好 (20-24 FPS)")
            elif avg_fps >= 15:
                print("性能评级: 一般 (15-19 FPS)")
            else:
                print("性能评级: 需要优化 (<15 FPS)")

            print("\n优化建议:")
            if avg_fps < 20:
                print("- 考虑进一步降低摄像头分辨率")
                print("- 增加人脸检测间隔（每4-6帧检测一次）")
                print("- 使用更小的饰品图像文件")

            if avg_detection_time > 50:
                print("- 人脸检测时间较长，建议降低检测精度")

            if avg_render_time > 30:
                print("- 渲染时间较长，建议优化饰品图像尺寸")

if __name__ == "__main__":
    test_performance()