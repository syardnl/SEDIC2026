import csv, json
from datetime import datetime
from pathlib import Path

FIELDS = [
    "timestamp", "frame_id", "class_name",
    "threat_level", "confidence", "x1", "y1", "x2", "y2"
]

class DetectionLogger:
    def __init__(self, out_path: str = "outputs/detection_log.csv"):
        self.path = Path(out_path)
        self.path.parent.mkdir(exist_ok=True)
        self._file = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        self._writer.writeheader()

    def log(self, frame_id: int, detections: list):
        ts = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            self._writer.writerow({
                "timestamp":    ts,
                "frame_id":     frame_id,
                "class_name":   d["class_name"],
                "threat_level": d["threat_level"],
                "confidence":   d["confidence"],
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })

    def close(self):
        self._file.close()

    def to_json(self) -> str:
        rows = []
        with open(self.path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return json.dumps(rows, indent=2)