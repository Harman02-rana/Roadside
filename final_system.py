"""Inference helpers for the existing RoadGuard YOLO models.

The trained weights and original rule-based risk thresholds are intentionally
kept intact. Models load on first use so importing this module is safe.
"""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
POTHOLE_WEIGHTS = ROOT / "runs" / "detect" / "train2" / "weights" / "best.pt"
GARBAGE_WEIGHTS = ROOT / "runs" / "detect" / "train3" / "weights" / "best.pt"


@lru_cache(maxsize=1)
def load_models():
    missing = [str(path) for path in (POTHOLE_WEIGHTS, GARBAGE_WEIGHTS) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Trained model weights not found: " + ", ".join(missing))
    return YOLO(str(POTHOLE_WEIGHTS)), YOLO(str(GARBAGE_WEIGHTS))


def check_illumination(image_path):
    """Return the existing low / medium / good illumination category."""
    return illumination_details(image_path)[0]


def illumination_details(image_path):
    """Return the category and mean grayscale brightness for an image path."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Could not read image for illumination analysis")
    brightness = float(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)))
    if brightness < 50:
        return "low", brightness
    if brightness < 120:
        return "medium", brightness
    return "good", brightness


def calculate_risk(potholes, garbage, light):
    """The project's original rule: 2 per pothole, 1 per waste item,
    +5 low light or +2 medium light; thresholds >10 and >5.
    """
    score = calculate_score(potholes, garbage, light)
    if score > 10:
        return "HIGH RISK"
    if score > 5:
        return "MEDIUM RISK"
    return "LOW RISK"


def calculate_score(potholes, garbage, light):
    score = potholes * 2 + garbage
    if light == "low":
        score += 5
    elif light == "medium":
        score += 2
    return score


def analyze_image(image_bytes):
    """Run both trained detectors and return results plus annotated RGB image."""
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    original = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError("The uploaded file is not a readable image")
    pothole_model, garbage_model = load_models()
    pothole_result = pothole_model(original, verbose=False)[0]
    garbage_result = garbage_model(original, verbose=False)[0]

    detections = []
    annotated = original.copy()
    for result, label, color in (
        (pothole_result, "Pothole", (255, 91, 99)),
        (garbage_result, "Waste", (80, 223, 208)),
    ):
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            confidence = float(box.conf[0].item())
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{label} {confidence:.0%}", (x1, max(22, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
            detections.append({"type": label, "confidence": confidence,
                               "box": [x1, y1, x2, y2]})

    potholes = sum(item["type"] == "Pothole" for item in detections)
    waste = sum(item["type"] == "Waste" for item in detections)
    brightness = float(np.mean(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)))
    light = "low" if brightness < 50 else "medium" if brightness < 120 else "good"
    score = calculate_score(potholes, waste, light)
    risk = calculate_risk(potholes, waste, light)
    return {
        "potholes": potholes, "waste": waste, "light": light,
        "brightness": brightness, "risk": risk, "score": score,
        "detections": detections,
        "annotated_rgb": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
    }


if __name__ == "__main__":
    sample = ROOT / "test_images" / "img-399.jpg"
    image = cv2.imread(str(sample))
    if image is None:
        raise SystemExit(f"Sample image not found: {sample}")
    result = analyze_image(cv2.imencode(".jpg", image)[1].tobytes())
    print("Potholes:", result["potholes"])
    print("Waste:", result["waste"])
    print("Illumination:", result["light"])
    print("FINAL RISK:", result["risk"], "(score:", result["score"], ")")
