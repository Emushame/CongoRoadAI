import argparse
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .core import Settings


def main():
    parser = argparse.ArgumentParser(description="CongoRoadAI — analyse locale de vidéos")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, help="Nouveau dossier de sortie (ne doit pas exister)")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--alpr", action="store_true")
    parser.add_argument("--threshold", type=int, default=3)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--horizon", type=float, default=0.42)
    parser.add_argument("--top-width", type=float, default=0.30)
    parser.add_argument("--bottom-width", type=float, default=0.90)
    parser.add_argument("--persistence", type=float, default=0.5)
    parser.add_argument("--alpr-every", type=int, default=5)
    args = parser.parse_args()
    settings = Settings(model=args.model, alpr=args.alpr, threshold=args.threshold,
                        confidence=args.confidence, horizon=args.horizon, top_width=args.top_width,
                        bottom_width=args.bottom_width, persistence=args.persistence, alpr_every=args.alpr_every)
    output = args.output or Path("outputs") / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6])
    try:
        from .engine import process_video
        summary = process_video(args.video, output, settings,
                                progress=lambda kind, data: print(data) if kind == "status" else None)
    except (Exception, KeyboardInterrupt) as exc:
        parser.exit(1, f"Analyse interrompue : {exc}\n")
    print(f"Terminé : {summary['frames']} images. Exports : {output.resolve()}")


if __name__ == "__main__":
    main()
