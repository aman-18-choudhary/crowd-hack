# modules/detection.py

import cv2
import numpy as np


class CrowdEstimator:
    def __init__(self):
        print("⏳ Initializing Crowd Density Estimator...")
        print("✅ Density Estimator Ready.")

    def estimate_density(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Smooth noise
        blur = cv2.GaussianBlur(gray, (15, 15), 0)

        # Adaptive threshold
        _, thresh = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Calculate density ratio
        occupied_pixels = np.sum(thresh == 255)
        total_pixels = thresh.shape[0] * thresh.shape[1]

        density_score = occupied_pixels / total_pixels

        return frame, density_score