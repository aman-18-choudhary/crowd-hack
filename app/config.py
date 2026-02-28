import cv2
import time
import numpy as np
import serial
import json
from datetime import datetime
from ultralytics import YOLO

from modules.prediction import CrowdPredictor

# ============================
# CONFIG
# ============================
MODEL_PATH = "models/lstm_model.keras"
SCALER_PATH = "models/scaler.pkl"
SERIAL_PORT = "/dev/cu.usbserial-5B150116871"  # Change if needed
BAUD_RATE = 115200

RISK_THRESHOLD = 0.75

# Shared status dictionary (used by API)
status_data = {
    "timestamp": None,
    "current_density": 0,
    "predicted_density": 0,
    "current_flow": 0,
    "predicted_flow": 0,
    "risk": False,
}

# ============================
# MAIN
# ============================
def main():
    global status_data

    # ----------------------------
    # SERIAL CONNECTION
    # ----------------------------
    try:
        esp = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print("✅ Connected to ESP32")
        time.sleep(2)
    except:
        esp = None
        print("⚠ ESP32 not connected (running without hardware)")

    # ----------------------------
    # CAMERA
    # ----------------------------
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera not accessible")
        return

    print("✅ Camera started successfully.")

    # ----------------------------
    # YOLO MODEL
    # ----------------------------
    print("⏳ Loading YOLO model...")
    yolo = YOLO("yolov8n.pt")
    print("✅ YOLO model loaded successfully.")

    # ----------------------------
    # LSTM PREDICTOR
    # ----------------------------
    predictor = CrowdPredictor()
    predictor.load()
    print("✅ LSTM model loaded successfully.")

    frame_buffer = []
    prev_count = 0
    prev_time = time.time()

    # ----------------------------
    # LOOP
    # ----------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # ----------------------------
        # PERSON DETECTION
        # ----------------------------
        results = yolo(frame)[0]
        person_count = 0

        for box in results.boxes:
            if int(box.cls[0]) == 0:  # Class 0 = person
                person_count += 1

        # ----------------------------
        # DENSITY + FLOW
        # ----------------------------
        frame_area = frame.shape[0] * frame.shape[1]
        density = person_count / (frame_area / 10000)

        current_time = time.time()
        flow = abs(person_count - prev_count) / (current_time - prev_time + 1e-6)

        prev_count = person_count
        prev_time = current_time

        # ----------------------------
        # BUFFER FOR LSTM
        # ----------------------------
        frame_buffer.append(density)

        if len(frame_buffer) > 10:
            frame_buffer.pop(0)

        if len(frame_buffer) == 10:
            predicted_density = predictor.predict(frame_buffer)
        else:
            predicted_density = density

        predicted_flow = flow

        # ----------------------------
        # RISK LOGIC
        # ----------------------------
        risk = predicted_density > RISK_THRESHOLD

        # ----------------------------
        # SEND TO ESP32
        # ----------------------------
        if esp:
            try:
                if risk:
                    esp.write(b"RISK_ON\n")
                else:
                    esp.write(b"RISK_OFF\n")
            except Exception as e:
                print("⚠ Serial write failed:", e)

        # ----------------------------
        # UPDATE STATUS (API USES THIS)
        # ----------------------------
        status_data = {
            "timestamp": datetime.now().isoformat(),
            "current_density": float(density),
            "predicted_density": float(predicted_density),
            "current_flow": float(flow),
            "predicted_flow": float(predicted_flow),
            "risk": risk,
        }

        # ----------------------------
        # DISPLAY
        # ----------------------------
        fps = 1 / (time.time() - start_time + 1e-6)

        cv2.putText(frame, f"People: {person_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.putText(frame, f"Density: {density:.2f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.putText(frame, f"Pred Density: {predicted_density:.2f}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        cv2.putText(frame, f"Flow: {flow:.2f}", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

        if risk:
            cv2.putText(frame, "⚠ RISK ALERT!", (20, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.putText(frame, f"FPS: {int(fps)}", (20, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("AI Crowd Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ----------------------------
    # CLEANUP
    # ----------------------------
    cap.release()
    cv2.destroyAllWindows()
    if esp:
        esp.close()


if __name__ == "__main__":
    main()