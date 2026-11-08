"""Explicit real-model smoke test (may download model weights).

Run: python -m scripts.check_models
This checks inference on a blank frame, not road/plate recognition accuracy.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from congoroad.core import Settings
from congoroad.engine import YoloDetector, create_alpr, plate_predictions


def main():
    frame = np.zeros((384, 640, 3), dtype=np.uint8)
    checks = [
        ("YOLO", lambda: YoloDetector(Settings()).predict(frame.copy())),
        ("FastALPR", lambda: plate_predictions(create_alpr(), frame.copy(), 0.7)),
    ]
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "input": "blank synthetic frame, 640x384",
              "accuracy_evaluated": False, "checks": {}}
    for name, check in checks:
        print(f"{name} : chargement et inférence…", flush=True)
        try:
            predictions = check()
            report["checks"][name] = {"passed": True, "detections": len(predictions)}
        except Exception as exc:
            report["checks"][name] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(report["checks"][name], ensure_ascii=False), flush=True)
    folder = Path("outputs/validation")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model-smoke.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if all(item["passed"] for item in report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
