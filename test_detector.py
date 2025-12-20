"""
测试轻量级检测器
"""

from visionlens.face_detector_lite import FaceDetectorLite
import cv2

def test_detector():
    print("测试轻量级人脸检测器...")

    try:
        # 初始化检测器
        detector = FaceDetectorLite(
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1
        )

        print(f"检测器类型: {detector.detector_type}")
        print(f"设备: {detector.get_device()}")

        # 初始化摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            return

        print("摄像头已打开，按 'q' 退出测试")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 检测人脸
            landmarks = detector.detect(frame)

            if landmarks is not None:
                # 在图像上绘制关键点
                for i, landmark in enumerate(landmarks):
                    x, y = int(landmark[0]), int(landmark[1])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                # 获取关键点
                key_points = detector.get_key_points(landmarks)
                if key_points:
                    for key, point in key_points.items():
                        x, y = int(point[0]), int(point[1])
                        cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
                        cv2.putText(frame, key, (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

                print(f"检测到关键点: {len(landmarks)} 个")
            else:
                print("未检测到人脸")

            cv2.imshow("Detector Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("测试完成")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_detector()