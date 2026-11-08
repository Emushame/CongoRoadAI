"""Desktop entry point: python app.py."""
import queue
import threading
import tkinter as tk
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from uuid import uuid4

from congoroad.core import Settings


class Application:
    def __init__(self, root):
        self.root = root
        root.title("CongoRoadAI · Laboratoire vidéo")
        root.geometry("1120x820")
        root.minsize(850, 650)
        root.configure(bg="#111b29")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#111b29")
        style.configure("TLabel", background="#111b29", foreground="#e1eaf5", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 25, "bold"))
        style.configure("TCheckbutton", background="#111b29", foreground="#e1eaf5")
        style.configure("TButton", padding=8, font=("Segoe UI", 10))
        self.path = tk.StringVar()
        self.status = tk.StringVar(value="Choisissez une vidéo locale pour commencer.")
        self.threshold = tk.StringVar(value="3")
        self.confidence = tk.StringVar(value="0.35")
        self.horizon = tk.StringVar(value="0.42")
        self.alpr = tk.BooleanVar(value=False)
        self.events = queue.Queue()
        self.latest = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.worker = None
        self.closing = False
        main = ttk.Frame(root, padding=24)
        main.pack(fill="both", expand=True)
        ttk.Label(main, text="CongoRoadAI", style="Title.TLabel").pack(anchor="w")
        ttk.Label(main, text="VISION PAR ORDINATEUR  /  ANALYSE LOCALE DE TRAJETS").pack(anchor="w", pady=(0, 18))
        row = ttk.Frame(main)
        row.pack(fill="x")
        ttk.Entry(row, textvariable=self.path).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Choisir une vidéo", command=self.choose).pack(side="left", padx=(12, 0))
        controls = ttk.Frame(main)
        controls.pack(fill="x", pady=16)
        for label, variable in [("Seuil véhicules", self.threshold), ("Confiance YOLO", self.confidence),
                                ("Horizon (0–1)", self.horizon)]:
            ttk.Label(controls, text=label).pack(side="left", padx=(0, 6))
            ttk.Entry(controls, textvariable=variable, width=6).pack(side="left", padx=(0, 18))
        ttk.Checkbutton(controls, text="Lire les plaques (FastALPR)", variable=self.alpr).pack(side="left")
        self.preview = tk.Label(main, text="Votre vidéo annotée apparaîtra ici\n\nVoitures · Motos · Bus · Camions",
                                bg="#0a111d", fg="#8ca4bf", font=("Segoe UI", 15))
        self.preview.pack(fill="both", expand=True)
        self.progress = ttk.Progressbar(main, mode="determinate")
        self.progress.pack(fill="x", pady=(15, 8))
        ttk.Label(main, textvariable=self.status, wraplength=1000).pack(anchor="w")
        actions = ttk.Frame(main)
        actions.pack(fill="x", pady=12)
        self.run_button = ttk.Button(actions, text="Analyser et exporter", command=self.start)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(actions, text="Arrêter", command=self.stop.set, state="disabled")
        self.stop_button.pack(side="left", padx=10)
        ttk.Label(main, text="Prototype expérimental • Alertes indicatives • Ne remplace pas l’attention du conducteur").pack(anchor="w")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.poll)

    def choose(self):
        path = filedialog.askopenfilename(filetypes=[("Vidéos", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("Tous les fichiers", "*.*")])
        if path:
            self.path.set(path)

    def emit(self, kind, payload):
        if kind == "frame":
            try:
                self.latest.get_nowait()
            except queue.Empty:
                pass
            self.latest.put_nowait(payload)
        else:
            self.events.put((kind, payload))

    def start(self):
        try:
            settings = replace(Settings(), threshold=int(self.threshold.get()),
                               confidence=float(self.confidence.get()), horizon=float(self.horizon.get()),
                               alpr=self.alpr.get())
            settings.validate()
            source = Path(self.path.get())
            if not source.is_file():
                raise ValueError("Choisissez une vidéo existante.")
        except ValueError as exc:
            messagebox.showerror("Paramètres", str(exc))
            return
        destination = Path(__file__).parent / "outputs" / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6])
        self.stop.clear()
        self.progress["value"] = 0
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status.set("Préparation du traitement…")

        def work():
            try:
                from congoroad.engine import process_video
                summary = process_video(source, destination, settings, self.stop, self.emit)
                self.emit("done", (destination, summary))
            except Exception as exc:
                self.emit("error", str(exc))
        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def poll(self):
        try:
            data = self.latest.get_nowait()
            from PIL import Image, ImageTk
            image = Image.fromarray(data["image"][:, :, ::-1])
            image.thumbnail((max(320, self.preview.winfo_width()), max(180, self.preview.winfo_height())))
            self.photo = ImageTk.PhotoImage(image)
            self.preview.configure(image=self.photo, text="")
            self.progress["value"] = data["current"] / max(1, data["total"]) * 100
            self.status.set(f"Image {data['current']}/{data['total'] or '?'} · {data['vehicles']} véhicule(s) dans la zone · " +
                            ("PRUDENCE : zone dense" if data["alert"] else "Analyse en cours"))
        except queue.Empty:
            pass
        while not self.events.empty():
            kind, payload = self.events.get_nowait()
            if kind == "status":
                self.status.set(payload)
            else:
                self.run_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
                if kind == "error":
                    self.status.set("Échec du traitement : " + payload)
                    if not self.closing:
                        messagebox.showerror("Analyse interrompue", payload)
                else:
                    folder, summary = payload
                    state = "Arrêté (export partiel)" if summary["cancelled"] else "Terminé"
                    self.status.set(f"{state} · {summary['frames']} images · Exports : {folder}")
        if self.closing and (self.worker is None or not self.worker.is_alive()):
            self.root.destroy()
            return
        self.root.after(100, self.poll)

    def close(self):
        self.closing = True
        self.stop.set()
        self.status.set("Fermeture après l’opération en cours et finalisation des exports…")


if __name__ == "__main__":
    root = tk.Tk()
    Application(root)
    root.mainloop()
