import asyncio
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import av
import cv2
import numpy as np

from congoroad.core import Detection, Settings
from congoroad.live import FrameAnalyzer
from congoroad.webcam import CameraProcessor
from congoroad.web_jobs import VideoJob, validate_video


class FakeModels:
    def __init__(self):
        self.ocr_calls = []

    def predict(self, frame, settings, read_plates):
        self.ocr_calls.append(read_plates)
        return [Detection((x, 120, x + 30, 200), "Voiture", .9) for x in (120, 150, 180)], []

    def detector_for(self, settings):
        owner = self

        class Detector:
            def predict(self, frame):
                return owner.predict(frame, settings, False)[0]
        return Detector()

    def alpr_adapter(self):
        raise AssertionError("OCR disabled for upload tests")


class LiveTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((240, 320, 3), dtype=np.uint8)

    def test_media_time_and_gap_reset(self):
        analyzer = FrameAnalyzer(Settings(persistence=.5), FakeModels())
        self.assertFalse(analyzer.process(self.frame, 0)[1]["alert"])
        self.assertTrue(analyzer.process(self.frame, .5)[1]["alert"])
        self.assertFalse(analyzer.process(self.frame, 5)[1]["alert"])
        self.assertTrue(analyzer.process(self.frame, 5.5)[1]["alert"])
        self.assertFalse(analyzer.process(self.frame, 0)[1]["alert"])

    def test_state_is_not_shared_between_visitors(self):
        models = FakeModels()
        first = FrameAnalyzer(Settings(), models)
        second = FrameAnalyzer(Settings(), models)
        first.process(self.frame, 0)
        self.assertTrue(first.process(self.frame, .5)[1]["alert"])
        self.assertFalse(second.process(self.frame, .5)[1]["alert"])

    def test_ocr_cadence_and_width(self):
        models = FakeModels()
        analyzer = FrameAnalyzer(Settings(alpr=True, alpr_every=3, max_width=480), models)
        for index in range(7):
            image, state = analyzer.process(np.zeros((720, 1280, 3), dtype=np.uint8), index / 10)
        self.assertEqual(image.shape[1], 480)
        self.assertEqual(models.ocr_calls, [True, False, False, True, False, False, True])

    def test_queued_frames_drop_backlog(self):
        processor = CameraProcessor(Settings(), FakeModels())
        frames = [av.VideoFrame.from_ndarray(self.frame, format="bgr24") for _ in range(20)]
        returned = asyncio.run(processor.recv_queued(frames))
        self.assertEqual(len(returned), 1)
        self.assertEqual(processor.snapshot()["processed"], 1)

    def test_inference_failure_is_visible_and_recovers(self):
        models = FakeModels()
        processor = CameraProcessor(Settings(), models)
        frame = av.VideoFrame.from_ndarray(self.frame, format="bgr24")
        with patch.object(models, "predict", side_effect=RuntimeError("modèle indisponible")):
            output = processor.recv(frame)
        self.assertIn("indisponible", processor.snapshot()["error"])
        self.assertGreater(output.to_ndarray(format="bgr24").max(), 0)
        processor.recv(frame)
        self.assertIsNone(processor.snapshot()["error"])


class WebUploadTests(unittest.TestCase):
    def video_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
            self.assertTrue(writer.isOpened())
            for _ in range(10):
                writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
            writer.release()
            return path.read_bytes()

    def test_upload_exports_playable_h264_and_cleans_up(self):
        job = VideoJob(self.video_bytes(), ".avi", Settings(), FakeModels())
        job.thread.join(timeout=30)
        state = job.snapshot()
        self.assertTrue(state["done"])
        self.assertIsNone(state["error"])
        self.assertEqual(state["summary"]["frames"], 10)
        self.assertTrue(state["video"])
        with av.open(io.BytesIO(state["video"])) as movie:
            self.assertEqual(movie.streams.video[0].codec_context.name, "h264")
            self.assertEqual(len(list(movie.decode(video=0))), 10)
        with zipfile.ZipFile(io.BytesIO(state["archive"])) as archive:
            self.assertEqual(set(archive.namelist()), {"video_annotee.mp4", "analyse.csv", "resume.json"})
            self.assertEqual(json.loads(archive.read("resume.json"))["source"], "Vidéo importée")
        folder = job.folder
        job.cleanup()
        self.assertFalse(folder.exists())

    def test_invalid_upload_rejected(self):
        with self.assertRaises(ValueError):
            VideoJob(b"invalid", ".mp4", Settings(), FakeModels())
        with self.assertRaises(ValueError):
            VideoJob(b"invalid", ".exe", Settings(), FakeModels())

    def test_duration_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.avi"
            path.write_bytes(self.video_bytes())
            with patch("congoroad.web_jobs.MAX_VIDEO_SECONDS", .1):
                with self.assertRaisesRegex(ValueError, "2 minutes"):
                    validate_video(path)


class StreamlitTests(unittest.TestCase):
    def test_first_render_and_camera_consent_gate(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=15).run()
        self.assertFalse(app.exception)
        self.assertTrue(next(b for b in app.button if b.label == "Analyser la vidéo").disabled)
        app.sidebar.radio[0].set_value("Caméra en direct").run()
        self.assertFalse(app.exception)
        self.assertTrue(next(b for b in app.button if b.label == "Préparer la caméra").disabled)
        app.checkbox(key="camera_consent").check().run()
        self.assertFalse(next(b for b in app.button if b.label == "Préparer la caméra").disabled)
