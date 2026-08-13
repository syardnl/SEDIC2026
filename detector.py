from ultralytics import YOLO
from utils.colours import THREAT_COLOURS, THREAT_LEVEL
import cv2, numpy as np

class GuardianDetector:
    def __init__(self, model_path: str, conf: float = 0.4):
        self.model = YOLO(model_path)
        self.conf = conf

    def predict(self, frame: np.ndarray) -> list[dict]:
        """
        Returns a list of detection dicts for one frame.
        Each dict: {class_name, confidence, bbox, colour, threat_level}
        """
        results = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "class_name":   cls_name,
                "confidence":   round(float(box.conf[0]), 3),
                "bbox":         (x1, y1, x2, y2),
                "colour":       THREAT_COLOURS.get(cls_name, (180, 180, 180)),
                "threat_level": THREAT_LEVEL.get(cls_name, "UNKNOWN"),
            })
        return detections

    def annotate(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Draws bounding boxes + labels onto a copy of the frame."""
        out = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            colour = d["colour"]           # BGR expected by cv2
            label  = f"{d['class_name']}  {d['confidence']:.0%}"
            threat = d["threat_level"]

            # Box
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

            # Label background pill
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), colour, -1)
            cv2.putText(out, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            # Threat badge (top-right of box)
            cv2.putText(out, threat, (x2 - 100, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
        return out