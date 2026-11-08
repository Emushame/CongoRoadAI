"""Bounded background uploads. No Streamlit calls from the worker thread."""
import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import zipfile

import cv2

from .engine import process_video

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_VIDEO_SECONDS = 120
MAX_JOB_SECONDS = 300


def validate_video(path):
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not capture.isOpened() or not 0 < fps < 1000 or not 0 < frames < 1e9:
            raise ValueError("Vidéo illisible ou durée inconnue. Essayez un fichier MP4 standard.")
        if frames / fps > MAX_VIDEO_SECONDS:
            raise ValueError("Cette démonstration accepte des extraits de 2 minutes maximum.")
    finally:
        capture.release()


def browser_video(source, destination):
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(source), "-an",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "25", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(destination)],
                   check=True, capture_output=True, timeout=120)


class VideoJob:
    def __init__(self, content, suffix, settings, models):
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError("La vidéo dépasse 100 Mo.")
        if suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv", ".m4v"}:
            raise ValueError("Format vidéo non pris en charge.")
        self.temp = tempfile.TemporaryDirectory(prefix="congoroad-")
        self.folder = Path(self.temp.name)
        self.source = self.folder / ("source" + suffix.lower())
        try:
            self.source.write_bytes(content)
            validate_video(self.source)
        except Exception:
            self.temp.cleanup()
            raise
        self.settings, self.models = settings, models
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.state = {"status": "Préparation…", "done": False, "error": None}
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def progress(self, kind, data):
        if time.monotonic() - self.started > MAX_JOB_SECONDS:
            self.stop.set()
            self.update(warning="Limite de calcul de 5 minutes atteinte : export partiel.")
        if kind == "frame":
            image = data["image"]
            h, w = image.shape[:2]
            preview = cv2.resize(image, (min(w, 640), max(2, int(h * min(w, 640) / w))))
            self.update(current=data["current"], total=data["total"], image=preview,
                        traffic=data.get("traffic"),
                        vehicles=data["vehicles"], alert=data["alert"], status="Analyse en cours…")
        elif kind == "status":
            self.update(status=data)

    def run(self):
        try:
            output = self.folder / "resultat"
            summary = process_video(self.source, output, self.settings, self.stop, self.progress,
                                    detector_factory=self.models.detector_for,
                                    alpr_factory=self.models.alpr_adapter)
            summary["source"] = "Vidéo importée"
            (output / "resume.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            web_video = None
            if summary["frames"]:
                self.update(status="Préparation de la lecture dans le navigateur…")
                try:
                    browser_video(output / "video_annotee.mp4", output / "video_web.mp4")
                    web_video = (output / "video_web.mp4").read_bytes()
                except (OSError, subprocess.SubprocessError, RuntimeError):
                    self.update(warning="L’aperçu web est indisponible ; le MP4 annoté reste téléchargeable.")
            archive = io.BytesIO()
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
                for name in ("video_annotee.mp4", "analyse.csv", "resume.json"):
                    path = output / name
                    if path.is_file():
                        bundle.write(path, name)
            self.update(status="Arrêté — export partiel" if summary["cancelled"] else "Analyse terminée",
                        summary=summary, video=web_video, archive=archive.getvalue())
        except Exception as exc:
            self.update(error=str(exc), status="Analyse interrompue")
        finally:
            self.update(done=True)

    def cleanup(self):
        if self.thread.is_alive():
            raise RuntimeError("Arrêtez le traitement avant de supprimer ses fichiers.")
        self.temp.cleanup()
