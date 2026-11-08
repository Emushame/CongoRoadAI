import threading
import time

import av
import cv2
from streamlit_webrtc import VideoProcessorBase

from .live import FrameAnalyzer


class CameraProcessor(VideoProcessorBase):
    def __init__(self, settings, models):
        self.analyzer = FrameAnalyzer(settings, models)
        self.processing_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.state = {}

    def snapshot(self):
        with self.state_lock:
            return dict(self.state)

    def recv(self, frame):
        with self.processing_lock:
            raw = frame.to_ndarray(format="bgr24")
            try:
                timestamp = float(frame.time) if frame.time is not None else time.monotonic()
                rendered, state = self.analyzer.process(raw, timestamp)
            except Exception as exc:
                rendered = raw.copy()
                cv2.rectangle(rendered, (0, 0), (rendered.shape[1], 60), (30, 30, 160), -1)
                cv2.putText(rendered, "Analyse indisponible", (12, 35), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
                state = {"error": str(exc), "updated_at": time.monotonic()}
            with self.state_lock:
                self.state = state
            return av.VideoFrame.from_ndarray(rendered, format="bgr24")

    async def recv_queued(self, frames):
        # Process the most recent frame instead of building latency under load.
        return [self.recv(frames[-1])]
