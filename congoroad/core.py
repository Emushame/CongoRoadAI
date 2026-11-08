from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    label: str
    confidence: float

    @property
    def anchor(self):
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2, y2)


@dataclass(frozen=True)
class Settings:
    model: str = "yolo11n.pt"
    confidence: float = 0.35
    threshold: int = 3
    persistence: float = 0.5
    horizon: float = 0.42
    top_width: float = 0.30
    bottom_width: float = 0.90
    alpr: bool = False
    alpr_every: int = 5
    ocr_confidence: float = 0.70
    max_width: int = 1280

    def validate(self):
        if not all(math.isfinite(value) for value in
                   (self.confidence, self.ocr_confidence, self.persistence, self.horizon,
                    self.top_width, self.bottom_width, self.threshold, self.alpr_every, self.max_width)):
            raise ValueError("Les paramètres doivent être des nombres finis.")
        if not 0 < self.confidence <= 1 or not 0 <= self.ocr_confidence <= 1:
            raise ValueError("Les confiances doivent être comprises entre 0 et 1.")
        if self.threshold < 1 or self.persistence < 0 or self.alpr_every < 1:
            raise ValueError("Seuil et intervalle positifs requis ; durée positive ou nulle.")
        if not 0 < self.horizon < 0.95:
            raise ValueError("L'horizon doit être compris entre 0 et 0,95.")
        if not 0 < self.top_width <= self.bottom_width <= 1:
            raise ValueError("Largeurs de zone invalides.")
        if self.max_width < 320:
            raise ValueError("Largeur minimale : 320 pixels.")


def polygon(width, height, settings):
    return [(int(width * (1 - settings.top_width) / 2), int(height * settings.horizon)),
            (int(width * (1 + settings.top_width) / 2), int(height * settings.horizon)),
            (int(width * (1 + settings.bottom_width) / 2), height - 1),
            (int(width * (1 - settings.bottom_width) / 2), height - 1)]


def in_zone(point, width, height, settings):
    x, y = point
    top = int(height * settings.horizon)
    if not top <= y <= height - 1:
        return False
    fraction = (y - top) / (height - 1 - top)
    half_width = width * (settings.top_width + fraction *
                          (settings.bottom_width - settings.top_width)) / 2
    return width / 2 - half_width <= x <= width / 2 + half_width


class DensityAlert:
    """Persistence measured in video time, independent of inference speed."""
    def __init__(self, settings):
        self.settings = settings
        self.since = None

    def update(self, count, timestamp):
        if count < self.settings.threshold:
            self.since = None
            return False
        if self.since is None:
            self.since = timestamp
        return timestamp - self.since + 1e-9 >= self.settings.persistence
