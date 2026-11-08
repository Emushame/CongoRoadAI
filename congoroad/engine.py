import csv
import json
import math
from pathlib import Path
from threading import Event
from time import perf_counter

import cv2
import numpy as np

from .core import Detection, DensityAlert, Settings, in_zone, polygon

LABELS = {2: ("Voiture", (90, 210, 90)), 3: ("Moto", (230, 180, 50)),
          5: ("Bus", (60, 180, 250)), 7: ("Camion", (210, 110, 220))}


class YoloDetector:
    def __init__(self, settings):
        from ultralytics import YOLO
        self.model = YOLO(settings.model)
        self.settings = settings

    def predict(self, frame, confidence=None):
        result = self.model.predict(frame, conf=self.settings.confidence if confidence is None else confidence,
                                    classes=list(LABELS), device="cpu", verbose=False)[0]
        return [Detection(tuple(box), LABELS[int(cls)][0], float(conf))
                for box, cls, conf in zip(result.boxes.xyxy.cpu().tolist(),
                                          result.boxes.cls.cpu().tolist(),
                                          result.boxes.conf.cpu().tolist())]


def create_alpr():
    from fast_alpr import ALPR
    return ALPR(detector_model="yolo-v9-t-384-license-plate-end2end",
                ocr_model="cct-xs-v2-global-model", ocr_device="cpu",
                detector_providers=["CPUExecutionProvider"])


def plate_predictions(model, frame, minimum):
    plates = []
    for result in model.predict(frame):
        box = result.detection.bounding_box
        confidence = float(np.mean(result.ocr.confidence)) if result.ocr else 0.0
        text = result.ocr.text if result.ocr and confidence >= minimum else "Illisible"
        plates.append({"box": [int(box.x1), int(box.y1), int(box.x2), int(box.y2)],
                       "text": text, "confidence": confidence})
    return plates


def annotate(frame, detections, selected, plates, active, settings, timestamp):
    height, width = frame.shape[:2]
    points = np.array(polygon(width, height, settings), dtype=np.int32)
    tint = frame.copy()
    color = (60, 60, 240) if active else (200, 170, 40)
    cv2.fillPoly(tint, [points], color)
    cv2.addWeighted(tint, 0.15, frame, 0.85, 0, frame)
    cv2.polylines(frame, [points], True, color, 2)
    colors = {name: c for name, c in LABELS.values()}
    for item in detections:
        x1, y1, x2, y2 = map(int, item.box)
        c = colors.get(item.label, (180, 180, 180))
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        cv2.putText(frame, f"{item.label} {item.confidence:.0%}", (x1, max(20, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
    for plate in plates:
        x1, y1, x2, y2 = plate["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, f"{plate['text']} ({plate['confidence']:.0%})", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    if active and selected:
        # Point to the most populated third of the configured ROI.
        groups = [[d for d in selected if min(2, int(d.anchor[0] / width * 3)) == i]
                  for i in range(3)]
        dense = max(groups, key=len)
        target = tuple(int(sum(d.anchor[j] for d in dense) / len(dense)) for j in (0, 1))
        cv2.arrowedLine(frame, (width // 2, max(90, int(height * 0.25))), target,
                       (30, 30, 255), 5, tipLength=0.18)
    cv2.rectangle(frame, (0, 0), (width, 72), (25, 30, 38), -1)
    status = "PRUDENCE - zone dense" if active else "Analyse indicative"
    cv2.putText(frame, f"{status} | Zone : {len(selected)} | {timestamp:.1f}s", (12, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, "PROTOTYPE EXPERIMENTAL - pas un dispositif de securite", (12, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
    return frame


def process_video(source, destination, settings=None, stop=None, progress=None,
                  detector_factory=YoloDetector, alpr_factory=create_alpr):
    settings = settings or Settings()
    settings.validate()
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_file():
        raise ValueError("Sélectionnez un fichier vidéo existant.")
    if destination == source or source.is_relative_to(destination):
        raise ValueError("Le dossier de sortie doit être distinct de la vidéo source.")
    # A new directory prevents accidental replacement of earlier analyses.
    destination.mkdir(parents=True, exist_ok=False)
    stop = stop or Event()
    notify = progress or (lambda *args: None)
    capture = cv2.VideoCapture(str(source))
    writer = None
    count = alerts = 0
    started = perf_counter()
    try:
        if not capture.isOpened():
            raise ValueError("Vidéo illisible ou codec non pris en charge.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("La cadence de cette vidéo est inconnue.")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = capture.read()
        if not ok:
            raise ValueError("La vidéo ne contient aucune image décodable.")
        notify("status", "Chargement de YOLO (téléchargement au premier lancement)…")
        detector = detector_factory(settings)
        alpr = None
        if settings.alpr and not stop.is_set():
            notify("status", "Chargement de FastALPR…")
            alpr = alpr_factory()
        alert = DensityAlert(settings)
        with (destination / "analyse.csv").open("w", newline="", encoding="utf-8-sig") as report:
            csv_writer = csv.writer(report)
            csv_writer.writerow(["image", "temps_s", "vehicules", "dans_zone", "alerte", "ocr_execute", "plaques_json"])
            while ok and not stop.is_set():
                h, w = frame.shape[:2]
                ratio = min(1, settings.max_width / w)
                size = (max(2, int(w * ratio) // 2 * 2), max(2, int(h * ratio) // 2 * 2))
                frame = cv2.resize(frame, size)
                if writer is None:
                    writer = cv2.VideoWriter(str(destination / "video_annotee.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
                    if not writer.isOpened():
                        raise RuntimeError("Impossible de créer la vidéo MP4.")
                detections = detector.predict(frame)
                selected = [d for d in detections if in_zone(d.anchor, *size, settings)]
                timestamp = count / fps
                active = alert.update(len(selected), timestamp)
                ocr_run = alpr is not None and count % settings.alpr_every == 0
                # Never reuse plate coordinates/text on a different frame.
                plates = plate_predictions(alpr, frame, settings.ocr_confidence) if ocr_run else []
                annotated = annotate(frame, detections, selected, plates, active, settings, timestamp)
                writer.write(annotated)
                csv_writer.writerow([count, round(timestamp, 4), len(detections), len(selected),
                                     int(active), int(ocr_run), json.dumps(plates, ensure_ascii=False)])
                count += 1
                alerts += int(active)
                notify("frame", {"image": annotated, "current": count, "total": total,
                                 "vehicles": len(selected), "alert": active})
                ok, frame = capture.read()
        if not stop.is_set() and total > 0 and count < total:
            raise RuntimeError(f"Décodage interrompu à {count}/{total} images. Les exports sont partiels.")
        summary = {"source": str(source), "frames": count, "fps": fps,
                   "alert_frames": alerts, "cancelled": stop.is_set(),
                   "processing_seconds": round(perf_counter() - started, 2),
                   "settings": vars(settings), "audio_preserved": False}
        (destination / "resume.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary
    except Exception as exc:
        (destination / "erreur.txt").write_text(f"Exports partiels, analyse incomplète.\n{exc}", encoding="utf-8")
        raise
    finally:
        capture.release()
        if writer is not None:
            writer.release()
