"""Optimized lightweight reflection + texture based liveness demo.

This file is an engineering-friendly replacement draft for the original reflection.py.
It keeps the original idea but adds:
- argparse parameters
- adaptive thresholding
- normalized contour-area threshold
- smaller GLCM gray levels for speed
- temporal smoothing
- module-safe main entry
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


@dataclass
class LivenessConfig:
    camera_id: int = 0
    min_face_size: int = 60
    reflect_area_ratio: float = 0.012
    glcm_levels: int = 32
    smooth_window: int = 7
    debug: bool = False


def detect_reflection(roi: np.ndarray, area_ratio: float) -> bool:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold is more robust than fixed threshold=70.
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        -5,
    )

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_area = max(roi.shape[0] * roi.shape[1], 1)
    max_area = max((cv2.contourArea(cnt) for cnt in contours), default=0)
    return (max_area / roi_area) > area_ratio


def quantize_gray(gray: np.ndarray, levels: int) -> np.ndarray:
    levels = max(8, min(levels, 256))
    scaled = (gray.astype(np.float32) / 256.0 * levels).astype(np.uint8)
    return np.clip(scaled, 0, levels - 1)


def analyze_texture(roi: np.ndarray, levels: int, debug: bool = False) -> bool:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
    quantized = quantize_gray(small, levels)

    glcm = graycomatrix(
        quantized,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels,
        symmetric=True,
        normed=True,
    )

    contrast = float(graycoprops(glcm, "contrast").mean())
    dissimilarity = float(graycoprops(glcm, "dissimilarity").mean())
    homogeneity = float(graycoprops(glcm, "homogeneity").mean())
    energy = float(graycoprops(glcm, "energy").mean())

    if debug:
        print(
            f"contrast={contrast:.3f}, dissimilarity={dissimilarity:.3f}, "
            f"homogeneity={homogeneity:.3f}, energy={energy:.3f}"
        )

    # These thresholds are normalized for 32-level GLCM, not the old 256-level GLCM.
    return 1.5 < contrast < 18.0 and 0.7 < dissimilarity < 4.5 and homogeneity > 0.20 and energy > 0.015


def is_alive(roi: np.ndarray, config: LivenessConfig) -> bool:
    return detect_reflection(roi, config.reflect_area_ratio) and analyze_texture(
        roi,
        levels=config.glcm_levels,
        debug=config.debug,
    )


def parse_args() -> LivenessConfig:
    parser = argparse.ArgumentParser(description="Lightweight liveness detection demo")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--min-face-size", type=int, default=60)
    parser.add_argument("--reflect-area-ratio", type=float, default=0.012)
    parser.add_argument("--glcm-levels", type=int, default=32)
    parser.add_argument("--smooth-window", type=int, default=7)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    return LivenessConfig(
        camera_id=args.camera,
        min_face_size=args.min_face_size,
        reflect_area_ratio=args.reflect_area_ratio,
        glcm_levels=args.glcm_levels,
        smooth_window=args.smooth_window,
        debug=args.debug,
    )


def main() -> None:
    config = parse_args()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(config.camera_id)
    history: deque[bool] = deque(maxlen=config.smooth_window)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {config.camera_id}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(config.min_face_size, config.min_face_size),
            )

            for x, y, w, h in faces:
                roi = frame[y : y + h, x : x + w]
                current_alive = is_alive(roi, config)
                history.append(current_alive)
                alive_ratio = sum(history) / max(len(history), 1)
                stable_alive = alive_ratio >= 0.6

                label = "Alive" if stable_alive else "Spoof"
                color = (0, 255, 0) if stable_alive else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Liveness", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
