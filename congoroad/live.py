"""State isolated per camera, with serialized access to shared CPU models."""
import math
import threading
import time

import cv2

from .core import DensityAlert, Settings, in_zone
from .engine import YoloDetector, annotate, create_alpr, plate_predictions
from .traffic import TrafficMonitor, draw_traffic


class SharedModels:
    def __init__(self, with_alpr=False):
        self.lock = threading.Lock()
        self.detector = YoloDetector(Settings())
        self.alpr = create_alpr() if with_alpr else None

    def predict(self, frame, settings, read_plates):
        with self.lock:
            detections = self.detector.predict(frame, confidence=settings.confidence)
            plates = plate_predictions(self.alpr, frame, settings.ocr_confidence) if read_plates and self.alpr else []
        return detections, plates

    def detector_for(self, settings):
        owner = self

        class Adapter:
            def predict(self, frame):
                return owner.predict(frame, settings, False)[0]
        return Adapter()

    def alpr_adapter(self):
        owner = self

        class Adapter:
            def predict(self, frame):
                with owner.lock:
                    return owner.alpr.predict(frame)
        return Adapter()


class FrameAnalyzer:
    def __init__(self, settings, models):
        settings.validate()
        self.settings = settings
        self.models = models
        self.alert = DensityAlert(settings)
        self.last_timestamp = None
        self.origin = None
        self.processed = 0
        self.traffic = TrafficMonitor(settings)

    def process(self, frame, timestamp):
        if not math.isfinite(timestamp):
            raise ValueError("Horodatage vidéo invalide.")
        started = time.monotonic()
        # A gap or a restarted camera must not count as continuous dense traffic.
        if self.last_timestamp is not None and (timestamp <= self.last_timestamp or timestamp - self.last_timestamp > 2):
            self.alert = DensityAlert(self.settings)
            self.origin = timestamp
        if self.origin is None:
            self.origin = timestamp
        self.last_timestamp = timestamp
        h, w = frame.shape[:2]
        ratio = min(1, self.settings.max_width / w)
        width, height = max(2, int(w * ratio) // 2 * 2), max(2, int(h * ratio) // 2 * 2)
        resized = cv2.resize(frame, (width, height))
        read_plates = self.settings.alpr and self.processed % self.settings.alpr_every == 0
        detections, plates = self.models.predict(resized, self.settings, read_plates)
        selected = [d for d in detections if in_zone(d.anchor, width, height, self.settings)]
        active = self.alert.update(len(selected), timestamp)
        traffic = self.traffic.update(selected, width, height, timestamp)
        rendered = annotate(resized, detections, selected, plates, active, self.settings, timestamp - self.origin)
        draw_traffic(rendered, traffic)
        self.processed += 1
        latency = time.monotonic() - started
        return rendered, {"processed": self.processed, "vehicles": len(selected), "alert": active,
                          "latency_ms": round(latency * 1000), "ocr_run": read_plates,
                          "updated_at": time.monotonic(), "error": None, "traffic": traffic}
