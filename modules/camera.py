# modules/camera.py

import cv2
import time


class Camera:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.prev_time = 0

    def start(self):
        # Try default backend first
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            # Try macOS backend explicitly
            self.cap = cv2.VideoCapture(
                self.camera_index, cv2.CAP_AVFOUNDATION
            )

        if not self.cap.isOpened():
            raise Exception("❌ Could not open camera.")

        # Give camera time to warm up
        time.sleep(1)

        print("✅ Camera started successfully.")

    def read_frame(self):
        ret, frame = self.cap.read()

        if not ret or frame is None:
            return None, 0

        current_time = time.time()
        fps = 1 / (current_time - self.prev_time) if self.prev_time else 0
        self.prev_time = current_time

        return frame, int(fps)

    def stop(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("🛑 Camera stopped.")