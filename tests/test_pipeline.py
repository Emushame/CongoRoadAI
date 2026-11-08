import csv
import tempfile
import unittest
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import cv2
import numpy as np

from congoroad.core import Detection, Settings
from congoroad.engine import plate_predictions, process_video


class FakeDetector:
    def __init__(self, settings):
        pass

    def predict(self, frame):
        return [Detection((x, 100, x + 30, 200), "Voiture", .9) for x in (120, 150, 180)]


class PipelineTests(unittest.TestCase):
    def decoded_frames(self, path):
        capture = cv2.VideoCapture(str(path))
        self.assertTrue(capture.isOpened())
        frames = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            capture.release()
        return frames

    def make_video(self, root):
        path = root / "source.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
        self.assertTrue(writer.isOpened())
        for _ in range(12):
            writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
        writer.release()
        return path

    def test_export_preserves_frames_and_alert_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_video(root)
            summary = process_video(source, root / "out", detector_factory=FakeDetector)
            self.assertEqual(summary["frames"], 12)
            self.assertEqual(summary["alert_frames"], 7)
            capture = cv2.VideoCapture(str(root / "out/video_annotee.mp4"))
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 12)
            self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 10)
            capture.release()
            frames = self.decoded_frames(root / "out/video_annotee.mp4")
            self.assertEqual(len(frames), 12)
            self.assertEqual(frames[0].shape, (240, 320, 3))
            self.assertGreater(int(frames[-1].max()), 0)
            with (root / "out/analyse.csv").open(encoding="utf-8-sig") as report:
                rows = list(csv.DictReader(report))
            self.assertEqual(len(rows), 12)
            self.assertEqual(rows[4]["alerte"], "0")
            self.assertEqual(rows[5]["alerte"], "1")
            self.assertEqual(rows[-1]['trafic_etat'], 'observing')
            self.assertIn('occupation_zone', rows[-1])
            self.assertEqual(sum(summary['traffic_frames'].values()), summary['frames'])
            with self.assertRaises(FileExistsError):
                process_video(source, root / "out", detector_factory=FakeDetector)

    def test_cancel_finalizes_partial_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_video(root)
            stop = Event()
            def callback(kind, value):
                if kind == "frame" and value["current"] == 3:
                    stop.set()
            summary = process_video(source, root / "out", stop=stop, progress=callback,
                                    detector_factory=FakeDetector)
            self.assertTrue(summary["cancelled"])
            self.assertEqual(summary["frames"], 3)
            self.assertEqual(len(self.decoded_frames(root / "out/video_annotee.mp4")), 3)

    def test_detector_failure_marks_partial_export_and_releases_writer(self):
        class FailingDetector(FakeDetector):
            calls = 0

            def predict(self, frame):
                self.calls += 1
                if self.calls == 3:
                    raise RuntimeError("Erreur simulée du détecteur")
                return super().predict(frame)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_video(root)
            with self.assertRaisesRegex(RuntimeError, "Erreur simulée"):
                process_video(source, root / "out", detector_factory=FailingDetector)
            self.assertTrue((root / "out/erreur.txt").exists())
            self.assertFalse((root / "out/resume.json").exists())
            self.assertEqual(len(self.decoded_frames(root / "out/video_annotee.mp4")), 2)

    def test_missing_source_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                process_video(root / "missing.mp4", root / "out", detector_factory=FakeDetector)
            self.assertFalse((root / "out").exists())

    def test_plate_missing_or_uncertain_ocr(self):
        box = SimpleNamespace(x1=1, y1=2, x2=30, y2=12)
        results = [SimpleNamespace(detection=SimpleNamespace(bounding_box=box), ocr=ocr)
                   for ocr in (None, SimpleNamespace(text="BAD", confidence=[.2, .4]),
                               SimpleNamespace(text="AB123", confidence=.95))]
        model = SimpleNamespace(predict=lambda frame: results)
        plates = plate_predictions(model, np.zeros((20, 40, 3)), .7)
        self.assertEqual([p["text"] for p in plates], ["Illisible", "Illisible", "AB123"])

    def test_ocr_cadence_does_not_reuse_old_plates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_video(root)
            calls = []
            def predict(frame):
                calls.append(1)
                return [SimpleNamespace(
                    detection=SimpleNamespace(bounding_box=SimpleNamespace(x1=130, y1=170, x2=155, y2=185)),
                    ocr=SimpleNamespace(text="AB123", confidence=.95))]
            process_video(source, root / "out", Settings(alpr=True, alpr_every=5),
                          detector_factory=FakeDetector,
                          alpr_factory=lambda: SimpleNamespace(predict=predict))
            self.assertEqual(len(calls), 3)
            with (root / "out/analyse.csv").open(encoding="utf-8-sig") as report:
                rows = list(csv.DictReader(report))
            self.assertIn("AB123", rows[0]["plaques_json"])
            self.assertEqual(rows[1]["plaques_json"], "[]")
            self.assertEqual(rows[1]["ocr_execute"], "0")

    def test_invalid_video_has_explicit_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "broken.mp4"
            source.write_bytes(b"invalid")
            with self.assertRaises(ValueError):
                process_video(source, root / "out", detector_factory=FakeDetector)
            self.assertTrue((root / "out/erreur.txt").exists())
