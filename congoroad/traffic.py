"""Explainable visual congestion indicators, not a trained traffic forecast."""
from collections import deque
import math

import cv2
import numpy as np

from .core import polygon


def overlap(a, b):
    x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = x * y
    return intersection / max(1e-9, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection)


class TrafficMonitor:
    def __init__(self, settings):
        self.settings = settings
        self.history = deque(maxlen=61)
        self.tracks = {}
        self.next_id = 1
        self.last = None
        self.dense_since = None

    def update(self, selected, width, height, timestamp):
        if not math.isfinite(timestamp):
            raise ValueError("Horodatage invalide")
        if self.last is not None and (timestamp <= self.last or timestamp-self.last > 2):
            self.__init__(self.settings)
        self.last = timestamp
        boxes = [tuple(v / (width if i % 2 == 0 else height) for i, v in enumerate(d.box)) for d in selected]
        self.tracks = {k: t for k, t in self.tracks.items() if timestamp-t['time'] <= 1}
        pairs = sorted([(overlap(t['box'], b), k, i) for k, t in self.tracks.items()
                        for i, b in enumerate(boxes) if t['label'] == selected[i].label], reverse=True)
        matched, used, current = set(), set(), []
        for score, k, i in pairs:
            if score < .2 or k in matched or i in used:
                continue
            t, b = self.tracks[k], boxes[i]
            dt = timestamp-t['time']
            movement = math.hypot((b[0]+b[2]-t['box'][0]-t['box'][2])/2, b[3]-t['box'][3]) / max(dt, 1e-6)
            weight = math.exp(-dt/.5)
            t.update(box=b, time=timestamp, movement=weight*t['movement']+(1-weight)*movement)
            matched.add(k)
            used.add(i)
            current.append(t)
        for i, b in enumerate(boxes):
            if i not in used:
                t = dict(box=b, label=selected[i].label, time=timestamp, start=timestamp, movement=1.0)
                self.tracks[self.next_id] = t
                self.next_id += 1
                current.append(t)
        # Union of boxes clipped to the ROI: overlapping detections never double its area.
        roi = np.zeros((100, 160), np.uint8)
        cv2.fillPoly(roi, [np.array(polygon(160, 100, self.settings), np.int32)], 1)
        occupied = np.zeros_like(roi)
        for b in boxes:
            x1, y1, x2, y2 = [int(max(0, min(1, v))* (160 if i % 2 == 0 else 100)) for i,v in enumerate(b)]
            occupied[y1:y2, x1:x2] = 1
        occupancy = float(np.count_nonzero(occupied & roi) / max(1, np.count_nonzero(roi)))
        count = len(selected)
        dense = count >= self.settings.threshold and occupancy >= .12
        self.dense_since = (self.dense_since if self.dense_since is not None else timestamp) if dense else None
        duration = timestamp-self.dense_since if self.dense_since is not None else 0
        if not self.history or timestamp-self.history[-1][0] >= .5-1e-9:
            self.history.append((timestamp, count, occupancy))
        while self.history and timestamp-self.history[0][0] > 30:
            self.history.popleft()
        recent = [r for r in self.history if timestamp-r[0] <= 10]
        age = timestamp-self.history[0][0]
        confirmed = [t for t in current if timestamp-t['start'] >= 2]
        slow = sum(t['movement'] < .01 for t in confirmed)
        delta = slope = occupancy_delta = 0.0
        if len(recent) >= 6 and recent[-1][0]-recent[0][0] >= 5:
            split = max(1, len(recent)//3)
            delta = float(np.mean([r[1] for r in recent[-split:]])-np.mean([r[1] for r in recent[:split]]))
            occupancy_delta = float(np.mean([r[2] for r in recent[-split:]])-np.mean([r[2] for r in recent[:split]]))
            times = np.array([r[0]-recent[0][0] for r in recent])
            slope = float(np.polyfit(times, [r[1] for r in recent], 1)[0])
        accumulation = delta >= 1 and occupancy_delta >= .03 and slope > .05
        level, label = 'observing', 'Observation en cours'
        reasons = ['Au moins 10 secondes de vidéo continue sont nécessaires.']
        if age >= 10:
            level, label = 'none', 'Pas de signal persistant'
            reasons = ['Aucun signal visuel persistant avec les seuils actuels ; cela ne garantit pas une route fluide.']
            if accumulation and count >= 2:
                level, label = 'watch', 'Accumulation en hausse'
                reasons = ['Le nombre de véhicules et la surface occupée augmentent sur les 10 dernières secondes.']
            if duration >= 8 and len(confirmed) >= 2:
                level, label = 'risk', 'Risque de congestion visuelle'
                reasons = ['Zone chargée depuis au moins 8 secondes, avec plusieurs véhicules suivis durablement.']
                if self.settings.camera_fixed and slow >= 2:
                    reasons.append('Au moins deux véhicules ont un faible mouvement apparent dans l’image.')
        return dict(level=level, label=label, reasons=reasons, occupancy=round(occupancy, 3),
                    tracked=len(confirmed), low_motion=slow if self.settings.camera_fixed else None,
                    dense_seconds=round(duration, 1), count_trend=round(delta, 2),
                    observation_seconds=round(age, 1),
                    history=[{'temps_s': round(r[0]-self.history[0][0],1), 'véhicules': r[1]} for r in self.history])


def draw_traffic(frame, state):
    labels = {'observing': 'Trafic : observation (10s)', 'none': 'Trafic : pas de signal persistant',
              'watch': 'Trafic : accumulation en hausse', 'risk': 'RISQUE DE CONGESTION - indicatif'}
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h-30), (w, h), (25,30,38), -1)
    color = (40,160,255) if state['level'] in ('watch','risk') else (230,230,230)
    cv2.putText(frame, labels[state['level']], (8,h-10), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
    return frame
