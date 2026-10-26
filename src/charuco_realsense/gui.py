"""A local Tk desktop UI for live recognition and jitter recording."""
import copy
from datetime import datetime
import math
from pathlib import Path
from queue import Empty

from .cli import _load_board
from .monitor import CaptureWorker


BG = "#eef2f6"
INK = "#182a3b"
MUTED = "#607386"
BLUE = "#2563eb"
GREEN = "#087f5b"
RED = "#c13c42"
AMBER = "#a76214"
LINE = "#dce4ed"
WHITE = "#ffffff"


def _number(value, unit, digits=3):
    return f"{value:.{digits}f} {unit}" if value is not None else f"-- {unit}"


def _nice_limit(value):
    if value <= 0:
        return 1.
    magnitude = 10. ** math.floor(math.log10(value))
    for multiple in (1, 2, 5, 10):
        if value <= multiple * magnitude:
            return multiple * magnitude
    return value


class StripChart:
    def __init__(self, parent, title, unit, series, window, symmetric=False):
        import tkinter as tk
        self.canvas = tk.Canvas(parent, bg=WHITE, highlightthickness=1,
                                highlightbackground=LINE, height=140)
        self.title, self.unit = title, unit
        self.series, self.window, self.symmetric = series, window, symmetric
        self.points = []
        self.canvas.bind("<Configure>", lambda event: self.draw())

    def set_points(self, points):
        self.points = points
        self.draw()

    def draw(self):
        c = self.canvas
        width, height = c.winfo_width(), c.winfo_height()
        if width < 100 or height < 80:
            return
        c.delete("all")
        c.create_text(14, 16, text=f"{self.title} · {self.unit}", anchor="w", fill=INK, font=("Helvetica", 9, "bold"))
        legend_x = width - 14
        for name, color, _ in reversed(self.series):
            c.create_text(legend_x, 16, text=name, anchor="e", fill=color, font=("Helvetica", 8))
            legend_x -= max(40, len(name) * 7 + 15)
        left, right, top, bottom = 62, width - 16, 38, height - 27
        values = [extract(p) for p in self.points for _, _, extract in self.series]
        values = [v for v in values if v is not None and math.isfinite(v)]
        limit = _nice_limit(max([abs(v) for v in values] + [.01]) * 1.12)
        lower, upper = (-limit, limit) if self.symmetric else (0., limit)
        for value in (lower, (lower + upper) / 2, upper):
            y = bottom - (value - lower) / (upper - lower) * (bottom - top)
            c.create_line(left, y, right, y, fill=LINE)
            c.create_text(left - 8, y, text=f"{value:.3g}", anchor="e", fill=MUTED, font=("Helvetica", 8))
        end = self.points[-1]["elapsed_s"] if self.points else 0.
        start = max(0., end - self.window)
        # Keep the initial history anchored at zero until the chart window fills.
        end_axis = max(self.window, end)
        start_axis = max(0., end_axis - self.window)
        for fraction in (0., .5, 1.):
            x = left + fraction * (right - left)
            label = start_axis + fraction * self.window
            c.create_text(x, height - 11, text=f"{label:.0f} s", fill=MUTED, font=("Helvetica", 8))
        for _, color, extract in self.series:
            segment = []
            for point in self.points:
                value = extract(point)
                if value is None or not math.isfinite(value):
                    if len(segment) >= 4:
                        c.create_line(*segment, fill=color, width=1.8)
                    segment = []
                    continue
                if point["elapsed_s"] < start:
                    continue
                x = left + (point["elapsed_s"] - start_axis) / self.window * (right - left)
                y = bottom - (value - lower) / (upper - lower) * (bottom - top)
                segment.extend((x, y))
            if len(segment) >= 4:
                c.create_line(*segment, fill=color, width=1.8)
            elif len(segment) == 2:
                x, y = segment
                c.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color, outline=color)
        if not values:
            c.create_text((left + right) / 2, (top + bottom) / 2, text="Waiting for measurements",
                          fill=MUTED, font=("Helvetica", 9))


class MonitorApp:
    def __init__(self, root, args, board_names, worker_factory=CaptureWorker, auto_connect=True):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.root, self.args = root, args
        self.worker_factory = worker_factory
        self.worker = None
        self.connected = self.recording = self.closing = False
        self.last_image = None
        self.photo = None
        self.last_session = None
        self.pending_record = False
        self.board_var = tk.StringVar(value=args.board)
        self.output_var = tk.StringVar(value=args.output_dir)
        self.save_frames_var = tk.BooleanVar(value=args.save_frames)
        self.status_var = tk.StringVar(value="Ready to connect")
        self.record_var = tk.StringVar(value="Preview only · press Start recording to save measurements")
        self.saved_var = tk.StringVar(value="Keep the camera and board stationary when measuring jitter.")
        self.device_var = tk.StringVar(value="Camera disconnected")
        self.recognition_var = tk.StringVar(value="Waiting for camera")
        self.ids_var = tk.StringVar(value="Matched --\nMissing --\nForeign --")
        self.quality_var = tk.StringVar(value="Pose quality will appear here.")
        self.metrics = {key: tk.StringVar(value="--") for key in ("markers", "corners", "position", "fit", "position_jitter", "rotation_jitter", "corner_jitter")}
        root.title("ChArUco monitor")
        root.geometry("1280x940")
        root.minsize(1040, 820)
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.option_add("*Font", ("Helvetica", 10))
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TButton", padding=(14, 8), font=("Helvetica", 10))
        style.configure("Primary.TButton", background=BLUE, foreground=WHITE)
        style.map("Primary.TButton", background=[("active", "#1d4ed8"), ("disabled", "#d4dfed")], foreground=[("disabled", "#8293a5")])
        style.configure("Stop.TButton", background=RED, foreground=WHITE)
        style.map("Stop.TButton", background=[("active", "#a62b32"), ("disabled", "#e6dadd")], foreground=[("disabled", "#a78d93")])
        style.configure("TCheckbutton", background=WHITE)
        self._build(board_names)
        self.root.after(50, self._poll)
        if auto_connect:
            self.root.after(150, self.connect)

    def _label(self, parent, text=None, variable=None, **options):
        return self.tk.Label(parent, text=text, textvariable=variable, bg=parent.cget("bg"),
                             fg=options.pop("fg", INK), **options)

    def _build(self, board_names):
        tk, ttk = self.tk, self.ttk
        root = self.root
        header = tk.Frame(root, bg=BG)
        header.pack(fill="x", padx=24, pady=(18, 12))
        self._label(header, "ChArUco monitor", font=("Helvetica", 21, "bold")).pack(side="left")
        self.status_badge = self._label(header, variable=self.status_var, fg=MUTED, font=("Helvetica", 10, "bold"))
        self.status_badge.pack(side="right")
        controls = tk.Frame(root, bg=WHITE, highlightbackground=LINE, highlightthickness=1)
        controls.pack(fill="x", padx=24, pady=(0, 12))
        first = tk.Frame(controls, bg=WHITE)
        first.pack(fill="x", padx=14, pady=(10, 8))
        self._label(first, "Board", fg=MUTED).pack(side="left", padx=(0, 10))
        self.board_box = ttk.Combobox(first, textvariable=self.board_var, values=board_names, state="readonly", width=13)
        self.board_box.pack(side="left", padx=(0, 12))
        self.connect_button = ttk.Button(first, text="Connect camera", command=self.connect)
        self.connect_button.pack(side="left")
        self.stop_button = ttk.Button(first, text="Stop & save", command=self.stop_recording, style="Stop.TButton", state="disabled")
        self.stop_button.pack(side="right")
        self.record_button = ttk.Button(first, text="Start recording", command=self.start_recording, style="Primary.TButton", state="disabled")
        self.record_button.pack(side="right", padx=10)
        self._label(first, variable=self.record_var, fg=MUTED, font=("Helvetica", 9)).pack(side="left", padx=16)
        second = tk.Frame(controls, bg=WHITE)
        second.pack(fill="x", padx=14, pady=(0, 10))
        self._label(second, "Save to", fg=MUTED).pack(side="left", padx=(0, 10))
        self.output_entry = ttk.Entry(second, textvariable=self.output_var)
        self.output_entry.pack(side="left", fill="x", expand=True)
        self.browse_button = ttk.Button(second, text="Browse...", command=self.browse)
        self.browse_button.pack(side="left", padx=8)
        self.save_check = ttk.Checkbutton(second, text="Also save PNG frames", variable=self.save_frames_var)
        self.save_check.pack(side="left")

        main = tk.Frame(root, bg=BG)
        main.pack(fill="both", expand=True, padx=24)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=0, minsize=310)
        main.rowconfigure(0, weight=3, minsize=300)
        main.rowconfigure(1, weight=2, minsize=285)
        preview_card = tk.Frame(main, bg=WHITE, highlightbackground=LINE, highlightthickness=1)
        preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=(0, 12))
        preview_heading = tk.Frame(preview_card, bg=WHITE)
        preview_heading.pack(fill="x", padx=14, pady=9)
        self._label(preview_heading, "LIVE RECOGNITION", font=("Helvetica", 9, "bold")).pack(side="left")
        self._label(preview_heading, variable=self.device_var, fg=MUTED, font=("Helvetica", 8)).pack(side="right")
        self.preview = tk.Canvas(preview_card, bg="#122231", highlightthickness=0)
        self.preview.pack(fill="both", expand=True)
        self.preview.bind("<Configure>", lambda event: self._render_image())
        self._label(preview_card, "Red outlines: marker IDs     |     Green points: ChArUco corners",
                    fg=MUTED, font=("Helvetica", 8), anchor="w").pack(fill="x", padx=14, pady=7)
        self.preview.create_text(200, 100, text="Connect a camera to begin", fill="#a3b4c6", tags="placeholder")
        stats = tk.Frame(main, bg=WHITE, highlightbackground=LINE, highlightthickness=1, padx=16, pady=12)
        stats.grid(row=0, column=1, sticky="nsew", pady=(0, 12))
        self.recognition_label = self._label(stats, variable=self.recognition_var, fg=MUTED, font=("Helvetica", 13, "bold"), anchor="w")
        self.recognition_label.pack(fill="x", pady=(0, 8))
        for caption, key in (("Matched markers", "markers"), ("ChArUco corners", "corners"),
                             ("Position X / Y / Z · mm", "position"), ("Image fit · RMS", "fit")):
            self._metric(stats, caption, key)
        self._label(stats, variable=self.ids_var, fg=MUTED, anchor="w", justify="left", wraplength=275,
                    font=("Helvetica", 8)).pack(fill="x", pady=(6, 8))
        self._label(stats, f"ROLLING JITTER · {self.args.rolling_seconds:g} s", fg=MUTED,
                    font=("Helvetica", 8, "bold"), anchor="w").pack(fill="x", pady=(0, 5))
        for caption, key in (("Position RMS", "position_jitter"), ("Rotation RMS", "rotation_jitter"), ("Corner RMS", "corner_jitter")):
            self._metric(stats, caption, key)
        self._label(stats, variable=self.quality_var, fg=MUTED, wraplength=275, justify="left", anchor="w",
                    font=("Helvetica", 8)).pack(fill="x", pady=(6, 0))
        charts = tk.Frame(main, bg=BG)
        charts.grid(row=1, column=0, columnspan=2, sticky="nsew")
        charts.columnconfigure((0, 1), weight=1)
        charts.rowconfigure((0, 1), weight=1)
        delta = lambda index: lambda p: p["delta_xyz_mm"][index] if p["delta_xyz_mm"] is not None else None
        window = self.args.window_seconds
        self.charts = [
            StripChart(charts, "Position change from first pose", "mm", [("X", BLUE, delta(0)), ("Y", GREEN, delta(1)), ("Z", AMBER, delta(2))], window, True),
            StripChart(charts, f"Position jitter · {self.args.rolling_seconds:g} s RMS", "mm", [("Position", BLUE, lambda p: p["position_jitter_mm"])], window),
            StripChart(charts, f"Rotation jitter · {self.args.rolling_seconds:g} s RMS", "deg", [("Rotation", "#8058c4", lambda p: p["rotation_jitter_deg"])], window),
            StripChart(charts, f"Corner jitter · {self.args.rolling_seconds:g} s RMS", "px", [("Corners", GREEN, lambda p: p["corner_jitter_px"])], window),
        ]
        for index, chart in enumerate(self.charts):
            chart.canvas.grid(row=index // 2, column=index % 2, sticky="nsew",
                              padx=(0, 6) if index % 2 == 0 else (6, 0), pady=(0, 8))
        self._label(root, variable=self.saved_var, fg=MUTED, anchor="w", justify="left", wraplength=1210,
                    font=("Helvetica", 9)).pack(fill="x", padx=24, pady=(6, 14))

    def _metric(self, parent, caption, key):
        row = self.tk.Frame(parent, bg=WHITE)
        row.pack(fill="x", pady=2)
        self._label(row, caption, fg=MUTED, font=("Helvetica", 8)).pack(side="left")
        self._label(row, variable=self.metrics[key], font=("Helvetica", 9, "bold")).pack(side="right")

    def browse(self):
        from tkinter import filedialog
        selected = filedialog.askdirectory(parent=self.root, title="Choose recording parent folder",
                                            initialdir=str(Path(self.output_var.get() or ".").expanduser().resolve()))
        if selected:
            self.output_var.set(selected)

    def connect(self):
        if self.closing:
            return
        if self.worker is not None and self.worker.is_alive():
            self.status_var.set("Disconnecting...")
            self.connect_button.configure(state="disabled")
            self.record_button.configure(state="disabled")
            self.stop_button.configure(state="disabled")
            self.worker.disconnect()
            return
        args = copy.copy(self.args)
        args.board = self.board_var.get()
        self.worker = self.worker_factory(args)
        self.connected = False
        self.last_image = None
        self._render_image()
        for chart in self.charts:
            chart.set_points([])
        self.status_var.set("Connecting & warming up...")
        self.status_badge.configure(fg=MUTED)
        self.recognition_var.set("Waiting for camera")
        self.recognition_label.configure(fg=MUTED)
        for variable in self.metrics.values():
            variable.set("--")
        self.ids_var.set("Matched --\nMissing --\nForeign --")
        self.quality_var.set("Pose quality will appear here.")
        self.board_box.configure(state="disabled")
        self.connect_button.configure(state="disabled")
        self.worker.start()

    def start_recording(self):
        if not self.connected or self.recording or self.pending_record or self.closing:
            return
        parent = self.output_var.get().strip()
        if not parent:
            self.saved_var.set("Choose a recording folder first.")
            return
        path = Path(parent).expanduser() / ("run-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        self.pending_record = True
        self.record_button.configure(state="disabled")
        self._record_options("disabled")
        self.record_var.set("Starting recording...")
        self.worker.start_recording(path, self.save_frames_var.get())

    def stop_recording(self):
        if self.recording and self.worker is not None:
            self.stop_button.configure(state="disabled")
            self.record_var.set("Saving recording...")
            self.worker.stop_recording()

    def _record_options(self, state):
        for widget in (self.output_entry, self.browse_button, self.save_check):
            widget.configure(state=state)

    def _event(self, kind, payload):
        if kind == "ready":
            self.connected = True
            self.status_var.set("Live")
            self.status_badge.configure(fg=GREEN)
            camera = payload["camera"]
            self.device_var.set(camera.get("stream_info", "Color camera"))
            self.connect_button.configure(text="Disconnect", state="normal")
            self.record_button.configure(state="normal")
        elif kind == "record_started":
            self.recording, self.pending_record = True, False
            self.last_session = payload
            self.stop_button.configure(state="normal")
            self.record_var.set("Recording")
            self.saved_var.set(f"Saving measurements to {payload}")
        elif kind == "record_stopped":
            self.recording = self.pending_record = False
            self.last_session = payload["path"]
            self.stop_button.configure(state="disabled")
            self.record_button.configure(state="normal" if self.connected else "disabled")
            self._record_options("normal")
            result = payload["summary"]
            self.record_var.set(f"Saved {result['frames_recorded']} frames")
            rate = result["rates"]["board_recognized"]
            prefix = "Partial recording saved" if result["status"] != "complete" else "Saved"
            message = f"{prefix}: {payload['path']}"
            message += f"  ·  Recognition {rate:.0%}" if rate is not None else " (no frames)"
            if result["error"]:
                message += f"  ·  {result['error']}"
            self.saved_var.set(message)
        elif kind in ("error", "record_error"):
            self.status_var.set("Recording error" if kind == "record_error" else "Camera error")
            self.status_badge.configure(fg=RED)
            self.saved_var.set(payload)
            if kind == "record_error":
                self.pending_record = False
                self.record_var.set("Recording did not start")
                self.record_button.configure(state="normal" if self.connected else "disabled")
                self._record_options("normal")
        elif kind == "disconnected":
            self.connected = self.recording = self.pending_record = False
            if self.status_var.get() not in ("Camera error", "Recording error"):
                self.status_var.set("Disconnected")
                self.status_badge.configure(fg=MUTED)
            self.device_var.set("Camera disconnected")
            self.recognition_var.set("Camera disconnected")
            self.recognition_label.configure(fg=MUTED)
            for variable in self.metrics.values():
                variable.set("--")
            self.last_image = None
            self._render_image()
            self.board_box.configure(state="readonly")
            self.connect_button.configure(text="Connect camera", state="normal")
            self.record_button.configure(state="disabled")
            self.stop_button.configure(state="disabled")
            self._record_options("normal")

    def _poll(self):
        if self.worker is not None:
            while True:
                try:
                    kind, payload = self.worker.events.get_nowait()
                except Empty:
                    break
                if not self.closing:
                    self._event(kind, payload)
            if not self.closing and self.connected:
                try:
                    preview = self.worker.previews.get_nowait()
                except Empty:
                    pass
                else:
                    self._show_preview(preview)
        if self.closing and (self.worker is None or not self.worker.is_alive()):
            self.root.destroy()
            return
        self.root.after(50, self._poll)

    def _show_preview(self, preview):
        report = preview["report"]
        self.last_image = preview["image"]
        self._render_image()
        recognized = report["board_recognized"]
        self.recognition_var.set("Board recognized" if recognized else "No matching board")
        self.recognition_label.configure(fg=GREEN if recognized else AMBER)
        self.metrics["markers"].set(f"{len(report['matched_marker_ids'])} / {len(report['expected_marker_ids'])}")
        self.metrics["corners"].set(f"{report['detected_charuco_corner_count']} / {report['expected_charuco_corner_count']}")
        self.ids_var.set("\n".join(f"{caption}  {', '.join(map(str, report[key])) or '--'}" for caption, key in
                                  (("Matched", "matched_marker_ids"), ("Missing", "missing_marker_ids"), ("Foreign", "foreign_marker_ids"))))
        pose = report["pose"]
        self.metrics["position"].set(" / ".join(f"{v:.1f}" for v in pose["position_mm"]) if pose else "--")
        self.metrics["fit"].set(_number(pose["reprojection_rmse_px"] if pose else None, "px", 2))
        point = report["live_jitter"]
        for key, source, unit in (("position_jitter", "position_jitter_mm", "mm"),
                                   ("rotation_jitter", "rotation_jitter_deg", "deg"),
                                   ("corner_jitter", "corner_jitter_px", "px")):
            self.metrics[key].set(_number(point[source], unit))
        if not recognized:
            self.quality_var.set("Place the selected board in view. Missing poses appear as gaps in the charts.")
        elif not pose:
            self.quality_var.set("Board found; not enough reliable corners for a pose.")
        elif report["pose_quality_passed"]:
            self.quality_var.set("Pose quality passed. Charts show raw measurements.")
        else:
            self.quality_var.set("Pose quality rejected: " + ", ".join(reason.replace("_", " ") for reason in report["pose_quality_reasons"]))
        if self.recording and preview["recording"]:
            seconds = int(preview["recording_seconds"])
            self.record_var.set(f"REC {seconds // 60:02d}:{seconds % 60:02d}  ·  {preview['recorded_frames']} frames")
        for chart in self.charts:
            chart.set_points(preview["points"])

    def _render_image(self):
        from PIL import Image, ImageTk
        self.preview.delete("all")
        width, height = self.preview.winfo_width(), self.preview.winfo_height()
        if self.last_image is None:
            self.preview.create_text(width / 2, height / 2, text="Connect a camera to begin", fill="#a3b4c6", font=("Helvetica", 12))
            return
        image = Image.fromarray(self.last_image[:, :, ::-1])
        image.thumbnail((max(1, width), max(1, height)), Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(image, master=self.root)
        self.preview.create_image(width / 2, height / 2, image=self.photo, anchor="center")

    def close(self):
        if self.closing:
            return
        self.closing = True
        self.status_var.set("Saving and closing camera...")
        for button in (self.connect_button, self.record_button, self.stop_button):
            button.configure(state="disabled")
        if self.worker is not None:
            self.worker.disconnect()


def run_gui(args):
    try:
        import tkinter as tk
        from PIL import ImageTk  # noqa: F401 -- fail before opening a window if missing
    except ImportError as exc:
        raise ValueError("Desktop UI needs Tkinter and Pillow. Install Python's Tk support and pip install -e '.[gui]'.") from exc
    _, _, manifest = _load_board(args.manifest, args.board, args.generator_path)
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise ValueError(f"Cannot open desktop UI: {exc}. Use a desktop display or the benchmark command.") from exc
    app = MonitorApp(root, args, [board["name"] for board in manifest["boards"]])
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.close()
        root.mainloop()
    return 0
