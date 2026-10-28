"""Desktop board tracker: camera-frame coordinates, unfiltered jitter and 24/7 health."""
import copy
from datetime import datetime
import logging
import math
import os
from pathlib import Path
from queue import Empty
import signal
import time

from .cli import _load_board
from .camera import CAMERA_MODELS
from .monitor import CaptureWorker, _free_bytes
from .service import (Watchdog, _give_up_waiting, configure_logging, default_log_file, describe_event, log,
                      recording_path)
from .shutdown import SignalStop


BG = "#f2f5f7"
INK = "#172f3c"
MUTED = "#465a67"  # Darker secondary text: core X fonts render without anti-aliasing.
BLUE = "#2864d7"
GREEN = "#087f72"
RED = "#be3b44"
AMBER = "#aa6318"
PURPLE = "#7855b3"
LINE = "#dde5e9"
WHITE = "#ffffff"
PREVIEW_BG = "#102734"
FONT = "Helvetica"
MONO = "Courier"
AXES = (("X", RED), ("Y", GREEN), ("Z", BLUE))
TONES = {"idle": "#7d8f99", "live": GREEN, "warn": AMBER, "error": RED, "rec": RED}
LEVEL_COLORS = {logging.INFO: INK, logging.WARNING: AMBER, logging.ERROR: RED}
STALE_SECONDS = 2.
GAP_SECONDS = 1.
EVENT_ROWS = 500
QUALITY_TEXT = {
    "nonpositive_depth": "Board is behind the camera in the estimated pose",
    "insufficient_corners": "Too few corners for the quality threshold",
    "insufficient_board_coverage": "Visible corners cover too little of the board",
    "insufficient_inlier_fraction": "Too many corners were rejected as outliers",
    "high_reprojection_rmse": "Reprojection RMS exceeds the threshold",
    "high_corner_error": "A corner exceeds the image-error threshold",
    "duplicate_expected_marker_ids": "An expected marker ID appears more than once",
    "invalid_corner_correspondences": "Corner IDs are ambiguous or invalid",
}


def _number(value, unit="", digits=3):
    text = f"{value:.{digits}f}" if value is not None else "--"
    return f"{text} {unit}".rstrip()


def _duration(seconds):
    seconds = int(max(0, seconds))
    days, seconds = divmod(seconds, 86400)
    text = f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
    return f"{days}d {text}" if days else text


def _gigabytes(value):
    return f"{value / 1e9:,.1f} GB" if value is not None else "--"


def _nice_limit(value):
    if value <= 0:
        return 1.
    magnitude = 10. ** math.floor(math.log10(value))
    return next(multiple * magnitude for multiple in (1, 2, 5, 10) if value <= multiple * magnitude)


def _choose_fonts(root):
    """Prefer a clean installed sans; a monospace only if it is a good one.

    Tk reports family names in varying case, and builds without Xft list only
    core X fonts. Helvetica digits are already tabular, so it is the fallback
    for both rather than a thin Courier.
    """
    global FONT, MONO
    from tkinter import font
    families = {name.lower(): name for name in font.families(root)}

    def pick(candidates):
        return next((families[name.lower()] for name in candidates if name.lower() in families), None)
    FONT = pick(("Inter", "Noto Sans", "Cantarell", "Ubuntu", "DejaVu Sans", "Liberation Sans", "Segoe UI",
                 "Helvetica", "Nimbus Sans L")) or "Helvetica"
    MONO = pick(("JetBrains Mono", "Noto Sans Mono", "Ubuntu Mono", "DejaVu Sans Mono", "Liberation Mono",
                 "Consolas", "Menlo")) or FONT


def _report_exception(kind, value, traceback):
    # A failing Tk callback must be visible but must never stop the monitor.
    log.error("UI callback failed", exc_info=(kind, value, traceback))


LABEL_FONT_FILES = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf")
LABEL_AXIS_COLORS = ((255, 120, 120), (90, 220, 160), (120, 170, 255))
_label_fonts = {}


def _label_font(size):
    """TrueType via Pillow: anti-aliased even when Tk itself only has core X fonts."""
    if size not in _label_fonts:
        from PIL import ImageFont
        for path in LABEL_FONT_FILES:
            try:
                _label_fonts[size] = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
        else:
            _label_fonts[size] = ImageFont.load_default(size)
    return _label_fonts[size]


def _overlaps(box, boxes):
    return any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in boxes)


def annotate_markers(image, markers, scale, selected=None, quality_passed=True):
    """Label each marker with its ID and camera-frame X/Y/Z in mm, at display resolution.

    Labels avoid each other and the other marker centres; one that cannot be
    placed shrinks to its ID. The selected marker is placed first and highlighted.
    Returns a new RGB image.
    """
    from PIL import Image, ImageDraw
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    title_font, body_font = _label_font(15), _label_font(13)
    centres = {id(m): (m["pixel_center"][0] * scale, m["pixel_center"][1] * scale) for m in markers}
    obstacles = [(x - 6, y - 6, x + 6, y + 6) for x, y in centres.values()]
    placed = []
    width, height = image.size
    for marker in sorted(markers, key=lambda m: (m["id"] != selected, m["id"])):
        cx, cy = centres[id(marker)]
        chosen = marker["id"] == selected
        position = marker["position_mm"]
        title = f"#{marker['id']}" + ("" if marker["matched"] else "  other board") + (
            "  duplicate" if marker["duplicate"] else "")
        lines = [(f"{axis}  {value:+.1f}", color) for axis, value, color in
                 zip("XYZ", position, LABEL_AXIS_COLORS)] if position else []
        title_w = draw.textlength(title, font=title_font)
        body_w = max((draw.textlength(text, font=body_font) for text, _ in lines), default=0)
        box_w, box_h = int(max(title_w, body_w)) + 14, 22 + 17 * len(lines) + 4
        box = None
        own = (cx - 6, cy - 6, cx + 6, cy + 6)
        others = [b for b in obstacles if b != own]
        for dx, dy in ((14, -box_h / 2), (-14 - box_w, -box_h / 2), (-box_w / 2, -14 - box_h), (-box_w / 2, 14)):
            x0 = min(max(2, cx + dx), width - box_w - 2)
            y0 = min(max(2, cy + dy), height - box_h - 2)
            candidate = (x0, y0, x0 + box_w, y0 + box_h)
            if not _overlaps(candidate, placed + others):
                box = candidate
                break
        if box is None and chosen:
            x0 = min(max(2, cx + 14), width - box_w - 2)
            y0 = min(max(2, cy - box_h / 2), height - box_h - 2)
            box = (x0, y0, x0 + box_w, y0 + box_h)
        outline = (255, 204, 0, 255) if chosen else (232, 160, 60, 255) if position and not quality_passed else (255, 255, 255, 90)
        if box is None:
            # No room for coordinates: a compact ID tag keeps every marker identifiable.
            text = f"#{marker['id']}"
            tw = draw.textlength(text, font=body_font)
            tag = (cx - tw / 2 - 5, cy - 9, cx + tw / 2 + 5, cy + 9)
            draw.rounded_rectangle(tag, 4, fill=(16, 39, 52, 200))
            draw.text((cx, cy), text, font=body_font, fill=(255, 255, 255, 255), anchor="mm")
            continue
        placed.append(box)
        edge_x = min(max(cx, box[0]), box[2])
        edge_y = min(max(cy, box[1]), box[3])
        if (edge_x, edge_y) != (cx, cy) and not (box[0] <= cx <= box[2] and box[1] <= cy <= box[3]):
            draw.line((cx, cy, edge_x, edge_y), fill=outline[:3] + (230,), width=2)
        draw.rounded_rectangle(box, 6, fill=(16, 39, 52, 228), outline=outline, width=3 if chosen else 1)
        draw.text((box[0] + 7, box[1] + 4), title, font=title_font, fill=(255, 255, 255, 255))
        for row, (text, color) in enumerate(lines):
            draw.text((box[0] + 7, box[1] + 24 + 17 * row), text, font=body_font, fill=color + (255,))
        radius = 6 if chosen else 4
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                     fill=(255, 204, 0, 255) if chosen else (255, 255, 255, 255), outline=(16, 39, 52, 255), width=2)
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


class StripChart:
    def __init__(self, parent, title, unit, series, window, symmetric=False):
        import tkinter as tk
        self.canvas = tk.Canvas(parent, bg=WHITE, highlightthickness=1,
                                highlightbackground=LINE, height=80)
        self.title, self.unit = title, unit
        self.series, self.window, self.symmetric = series, window, symmetric
        self.points = []
        self.canvas.bind("<Configure>", lambda event: self.draw())

    def set_points(self, points):
        self.points = points
        if self.canvas.winfo_viewable():
            self.draw()

    def _latest(self, extract):
        for point in reversed(self.points[-5:]):
            value = extract(point)
            if value is not None and math.isfinite(value):
                return value
        return None

    def draw(self):
        c = self.canvas
        width, height = c.winfo_width(), c.winfo_height()
        if width < 100 or height < 70:
            return
        c.delete("all")
        c.create_text(14, 15, text=f"{self.title} / {self.unit}", anchor="w", fill=INK, font=(FONT, 11, "bold"))
        # Latest values double as the legend, so the current reading is always visible.
        legend_x = width - 14
        for name, color, extract in reversed(self.series):
            latest = self._latest(extract)
            if name:
                text = f"{name} {latest:+.3f}" if latest is not None else f"{name} --"
            else:
                text = f"{latest:.3f}" if latest is not None else "--"
            item = c.create_text(legend_x, 15, text=text, anchor="e", fill=color, font=(MONO, 11, "bold"))
            bounds = c.bbox(item)
            legend_x -= (bounds[2] - bounds[0] if bounds else 60) + 14
        left, right, top, bottom = 58, width - 14, 32, height - 22
        end = self.points[-1]["elapsed_s"] if self.points else 0.
        start = max(0., end - self.window)
        points = [p for p in self.points if p["elapsed_s"] >= start]
        values = [extract(p) for p in points for _, _, extract in self.series]
        values = [v for v in values if v is not None and math.isfinite(v)]
        limit = _nice_limit(max([abs(v) for v in values] + [.01]) * 1.12)
        lower, upper = (-limit, limit) if self.symmetric else (0., limit)
        for value in (lower, (lower + upper) / 2, upper):
            y = bottom - (value - lower) / (upper - lower) * (bottom - top)
            c.create_line(left, y, right, y, fill=LINE)
            c.create_text(left - 8, y, text=f"{value:.3g}", anchor="e", fill=MUTED, font=(FONT, 10))
        start_axis = max(0., max(self.window, end) - self.window)
        for fraction, anchor in ((0., "w"), (.5, "center"), (1., "e")):
            x = left + fraction * (right - left)
            c.create_text(x, height - 9, text=_duration(start_axis + fraction * self.window)
                          if self.window >= 3600 else f"{start_axis + fraction * self.window:.0f} s",
                          anchor=anchor, fill=MUTED, font=(FONT, 10))
        for _, color, extract in self.series:
            segment = []
            previous = None
            for point in points:
                value = extract(point)
                # Missing values and time gaps (no frames, e.g. a camera outage) break the line.
                gap = previous is not None and point["elapsed_s"] - previous > GAP_SECONDS
                previous = point["elapsed_s"]
                if gap or value is None or not math.isfinite(value):
                    self._segment(segment, color)
                    segment = []
                if value is None or not math.isfinite(value):
                    continue
                x = left + (point["elapsed_s"] - start_axis) / self.window * (right - left)
                y = bottom - (value - lower) / (upper - lower) * (bottom - top)
                segment.extend((x, y))
            self._segment(segment, color)
        if not values:
            c.create_text((left + right) / 2, (top + bottom) / 2, text="Waiting for measurements",
                          fill=MUTED, font=(FONT, 11))

    def _segment(self, segment, color):
        if len(segment) >= 4:
            self.canvas.create_line(*segment, fill=color, width=1.8)
        elif len(segment) == 2:
            x, y = segment
            self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color, outline=color)


class MonitorApp:
    def __init__(self, root, args, board_names, worker_factory=CaptureWorker, auto_connect=True, shutdown=None,
                 log_path=None):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.root, self.args, self.shutdown = root, args, shutdown
        self.worker_factory = worker_factory
        self.worker = None
        self.connected = self.recording = self.closing = self.disconnecting = False
        self.reconnecting = self.destroyed = self.abandoned = False
        self.last_image = self.photo = self.last_report = self.last_session = None
        self.image_item = None
        self.pending_record = False
        self.close_started = None
        self.started = time.monotonic()
        self.last_frame_at = None
        self.retry_at = None
        self.lost_error = None
        self.lost_attempt = 0
        self.health = {}
        self.record_part = None
        self.record_seconds = 0.
        self.selected_marker = None
        self._display = None  # (x offset, y offset, scale) of the preview image on its canvas
        self.stall_reported = False
        self._overlay_text = None
        self._next_disk_check = 0.
        self._last_countdown = None
        self.watchdog = Watchdog(getattr(args, "stall_seconds", 30.))
        self.log_path = log_path
        _choose_fonts(root)
        self.board_var = tk.StringVar(value=args.board)
        self.camera_model_var = tk.StringVar(value=getattr(args, "camera_model", "auto"))
        self.output_var = tk.StringVar(value=args.output_dir)
        self.save_frames_var = tk.BooleanVar(value=args.save_frames)
        self.status_var = tk.StringVar(value="Disconnected")
        self.rec_pill_var = tk.StringVar(value="")
        self.record_var = tk.StringVar(value="Preview only")
        self.saved_var = tk.StringVar(value="Keep the camera and board still to measure jitter. "
                                            "Start recording (Ctrl+R) to save coordinates and measurements.")
        self.health_var = tk.StringVar(value="")
        self.marker_labels_var = tk.BooleanVar(value=True)
        self.position_title_var = tk.StringVar(value="BOARD ORIGIN")
        self.position_note_var = tk.StringVar(value="Click a marker to show its centre. X right / Y down / Z forward.")
        self.disk_var = tk.StringVar(value="")
        self.device_var = tk.StringVar(value="No camera connected")
        self.recognition_var = tk.StringVar(value="Waiting for camera")
        self.ids_var = tk.StringVar(value="Marker IDs will appear when a camera is connected.")
        self.quality_var = tk.StringVar(value="Place the selected board in view.")
        self.pose_state_var = tk.StringVar(value="No pose available")
        self.performance_var = tk.StringVar(value="No frames received")
        self.coordinate_note_var = tk.StringVar(value="Marker X/Y/Z are inferred from the board pose and geometry; they are not depth readings.")
        keys = ("markers", "corners", "fit", "position_jitter", "rotation_jitter", "corner_jitter", "x", "y", "z")
        self.metrics = {key: tk.StringVar(value="--") for key in keys}
        root.title("RealSense | ChArUco board tracker")
        screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{max(1200, min(1440, screen_w - 40))}x{max(800, min(960, screen_h - 80))}")
        root.minsize(1200, 800)
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self.close)
        # A plain function: a bound method here would make root and app a reference cycle.
        root.report_callback_exception = _report_exception
        root.bind("<Control-q>", lambda event: self.close())
        root.bind("<Control-r>", lambda event: self.toggle_recording())
        root.bind("<F11>", lambda event: self._fullscreen(not int(root.attributes("-fullscreen"))))
        root.bind("<Escape>", lambda event: self._fullscreen(False))
        root.option_add("*Font", (FONT, 11))
        self._style()
        self._build(board_names)
        self._log(logging.INFO, f"Monitor started for board {args.board!r}"
                  + (f"; log file {log_path}" if log_path else ""))
        self.watchdog.ready("Monitor open")
        self.root.after(50, self._poll)
        if auto_connect:
            self.root.after(150, self.connect)

    def _style(self):
        style = self.ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TButton", padding=(13, 6), font=(FONT, 11), background=WHITE, foreground=INK, bordercolor=LINE)
        style.configure("Primary.TButton", background=GREEN, foreground=WHITE, bordercolor=GREEN)
        style.map("Primary.TButton", background=[("active", "#06665c"), ("disabled", "#d7e3e3")], foreground=[("disabled", "#8b9d9e")])
        style.configure("Stop.TButton", background=RED, foreground=WHITE, bordercolor=RED)
        style.map("Stop.TButton", background=[("active", "#9e3038"), ("disabled", "#e9dfe0")], foreground=[("disabled", "#aa9295")])
        style.configure("Small.TButton", padding=(10, 4), font=(FONT, 11))
        style.configure("TCheckbutton", background=WHITE, foreground=MUTED, font=(FONT, 11))
        style.configure("Heading.TCheckbutton", background=WHITE, foreground=INK, font=(FONT, 11))
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(16, 6), font=(FONT, 11), background=BG, foreground=MUTED)
        style.map("TNotebook.Tab", background=[("selected", WHITE)], foreground=[("selected", INK)])
        style.configure("Treeview", background=WHITE, fieldbackground=WHITE, foreground=INK,
                        rowheight=30, font=(FONT, 11), borderwidth=0)
        style.configure("Treeview.Heading", background=BG, foreground=MUTED, font=(FONT, 11, "bold"), padding=(8, 6))

    def _label(self, parent, text=None, variable=None, **options):
        return self.tk.Label(parent, text=text, textvariable=variable, bg=parent.cget("bg"),
                             fg=options.pop("fg", INK), **options)

    def _card(self, parent, **options):
        return self.tk.Frame(parent, bg=WHITE, highlightbackground=LINE, highlightthickness=1, **options)

    def _pill(self, parent, variable):
        return self.tk.Label(parent, textvariable=variable, bg=TONES["idle"], fg=WHITE,
                             font=(FONT, 11, "bold"), padx=10, pady=3)

    def _build(self, board_names):
        tk, ttk = self.tk, self.ttk
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=20, pady=(10, 8))
        self._label(header, "Board tracker", font=(FONT, 18, "bold")).pack(side="left")
        self._label(header, "RealSense / ChArUco", fg=GREEN, font=(FONT, 11, "bold")).pack(side="left", padx=(12, 0), pady=(6, 0))
        ttk.Button(header, text="Quit", style="Small.TButton", command=self.close).pack(side="right")
        self.status_badge = self._pill(header, self.status_var)
        self.status_badge.pack(side="right", padx=(8, 12))
        self.rec_pill = self._pill(header, self.rec_pill_var)
        self.rec_pill.configure(bg=RED)

        controls = self._card(self.root)
        controls.pack(fill="x", padx=20, pady=(0, 10))
        first = tk.Frame(controls, bg=WHITE)
        first.pack(fill="x", padx=14, pady=(9, 7))
        self._label(first, "Camera model", fg=MUTED).pack(side="left", padx=(0, 9))
        self.camera_model_box = ttk.Combobox(first, textvariable=self.camera_model_var, values=CAMERA_MODELS,
                                            state="readonly", width=8)
        self.camera_model_box.pack(side="left", padx=(0, 14))
        self._label(first, "Board", fg=MUTED).pack(side="left", padx=(0, 9))
        self.board_box = ttk.Combobox(first, textvariable=self.board_var, values=board_names, state="readonly", width=12)
        self.board_box.pack(side="left", padx=(0, 10))
        self.connect_button = ttk.Button(first, text="Connect camera", command=self.connect)
        self.connect_button.pack(side="left")
        self.stop_button = ttk.Button(first, text="Stop & save", command=self.stop_recording, style="Stop.TButton", state="disabled")
        self.stop_button.pack(side="right")
        self.record_button = ttk.Button(first, text="Start recording", command=self.start_recording, style="Primary.TButton", state="disabled")
        self.record_button.pack(side="right", padx=8)
        self.record_label = self._label(first, variable=self.record_var, fg=MUTED, font=(FONT, 11))
        self.record_label.pack(side="left", padx=16)
        second = tk.Frame(controls, bg=WHITE)
        second.pack(fill="x", padx=14, pady=(0, 9))
        self._label(second, "Recordings", fg=MUTED, font=(FONT, 11)).pack(side="left", padx=(0, 10))
        self.output_entry = ttk.Entry(second, textvariable=self.output_var)
        self.output_entry.pack(side="left", fill="x", expand=True)
        self.disk_label = self._label(second, variable=self.disk_var, fg=MUTED, font=(FONT, 11))
        self.disk_label.pack(side="left", padx=(10, 0))
        self.browse_button = ttk.Button(second, text="Choose folder...", style="Small.TButton", command=self.browse)
        self.browse_button.pack(side="left", padx=8)
        self.save_check = ttk.Checkbutton(second, text="Save raw images (PNG)", variable=self.save_frames_var)
        self.save_check.pack(side="left")

        # The status bar is packed before the expanding area so it survives compact sizes.
        statusbar = tk.Frame(self.root, bg=BG)
        statusbar.pack(side="bottom", fill="x", padx=20, pady=(6, 9))
        self.footer_label = self._label(statusbar, variable=self.saved_var, fg=INK, anchor="w", justify="left",
                                        font=(FONT, 11), wraplength=1280)
        self.footer_label.pack(fill="x")
        self._label(statusbar, variable=self.health_var, fg=MUTED, anchor="w", font=(FONT, 10)).pack(fill="x", pady=(2, 0))
        statusbar.bind("<Configure>", lambda event: self.footer_label.configure(wraplength=max(100, event.width)))

        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=20)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=0, minsize=430)
        main.rowconfigure(0, weight=3, minsize=350)
        main.rowconfigure(1, weight=2, minsize=180)
        self._build_preview(main)
        self._build_stats(main)

        self.notebook = ttk.Notebook(main)
        self.notebook.grid(row=1, column=0, columnspan=2, sticky="nsew")
        charts = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(charts, text="Jitter over time")
        charts.columnconfigure((0, 1), weight=1)
        charts.rowconfigure((0, 1), weight=1)
        delta = lambda index: lambda p: p["delta_xyz_mm"][index] if p["delta_xyz_mm"] is not None else None
        window = self.args.window_seconds
        self.charts = [
            StripChart(charts, "Displacement from first pose", "mm", [(axis, color, delta(i)) for i, (axis, color) in enumerate(AXES)], window, True),
            StripChart(charts, "Position jitter", "mm RMS", [("", GREEN, lambda p: p["position_jitter_mm"])], window),
            StripChart(charts, "Rotation jitter", "deg RMS", [("", PURPLE, lambda p: p["rotation_jitter_deg"])], window),
            StripChart(charts, "Corner jitter", "px RMS", [("", BLUE, lambda p: p["corner_jitter_px"])], window),
        ]
        for index, chart in enumerate(self.charts):
            chart.canvas.grid(row=index // 2, column=index % 2, sticky="nsew",
                              padx=(0, 5) if index % 2 == 0 else (5, 0), pady=(6, 0))
        self._build_table()
        self._build_event_log()
        self.notebook.bind("<<NotebookTabChanged>>", lambda event: self._redraw_charts())

    def _build_preview(self, main):
        tk = self.tk
        preview_card = self._card(main)
        preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        preview_heading = tk.Frame(preview_card, bg=WHITE)
        preview_heading.pack(fill="x", padx=14, pady=7)
        self._label(preview_heading, "LIVE VIEW", font=(FONT, 11, "bold")).pack(side="left")
        self.ttk.Checkbutton(preview_heading, text="Coordinates on markers", variable=self.marker_labels_var,
                             style="Heading.TCheckbutton", command=self._render_image).pack(side="left", padx=(16, 0))
        self._label(preview_heading, variable=self.device_var, fg=MUTED, font=(FONT, 10)).pack(side="right")
        preview_footer = tk.Frame(preview_card, bg=WHITE)
        preview_footer.pack(side="bottom", fill="x")
        self._label(preview_footer, variable=self.performance_var, fg=MUTED, font=(FONT, 10), anchor="w").pack(side="left", padx=14, pady=5)
        self._label(preview_footer, "Click a marker to select it  /  mm, camera frame",
                    fg=MUTED, font=(FONT, 10), anchor="e").pack(side="right", padx=14, pady=5)
        self.preview = tk.Canvas(preview_card, bg=PREVIEW_BG, highlightthickness=0)
        self.preview.pack(fill="both", expand=True)
        self.preview.bind("<Configure>", lambda event: self._render_image())
        self.preview.bind("<Button-1>", self._on_preview_click)

    def _build_stats(self, main):
        tk, ttk = self.tk, self.ttk
        stats = self._card(main, padx=14, pady=8)
        stats.grid(row=0, column=1, sticky="nsew", pady=(0, 10))
        heading = tk.Frame(stats, bg=WHITE)
        heading.pack(fill="x")
        self._label(heading, variable=self.position_title_var, font=(FONT, 11, "bold")).pack(side="left")
        self._label(heading, "mm, camera frame", fg=MUTED, font=(FONT, 11)).pack(side="left", padx=(8, 0))
        self.copy_button = ttk.Button(heading, text="Copy XYZ", command=self.copy_coordinates, style="Small.TButton", state="disabled")
        self.copy_button.pack(side="right")
        positions = tk.Frame(stats, bg=WHITE)
        positions.pack(fill="x", pady=(6, 0))
        for index, (axis, color) in enumerate(AXES):
            positions.columnconfigure(index, weight=1, uniform="axis")
            cell = tk.Frame(positions, bg=BG, padx=8, pady=5)
            cell.grid(row=0, column=index, sticky="ew", padx=(0, 6) if index < 2 else 0)
            self._label(cell, axis, fg=color, font=(FONT, 11, "bold"), anchor="w").pack(fill="x")
            self._label(cell, variable=self.metrics[axis.lower()], font=(MONO, 24, "bold"), anchor="e").pack(fill="x")
        self._label(stats, variable=self.position_note_var, fg=MUTED, font=(FONT, 10), anchor="w",
                    justify="left", wraplength=400).pack(fill="x", pady=(4, 4))
        self.pose_state_label = self._label(stats, variable=self.pose_state_var, fg=MUTED, font=(FONT, 11, "bold"), anchor="w")
        self.pose_state_label.pack(fill="x")
        self._label(stats, variable=self.quality_var, fg=MUTED, font=(FONT, 10), wraplength=400,
                    justify="left", anchor="w").pack(fill="x", pady=(1, 4))
        recognition = tk.Frame(stats, bg=WHITE)
        recognition.pack(fill="x")
        self.recognition_label = self._label(recognition, variable=self.recognition_var, fg=MUTED, font=(FONT, 11, "bold"), anchor="w")
        self.recognition_label.pack(side="left")
        self._label(recognition, variable=self.metrics["fit"], fg=MUTED, font=(FONT, 11)).pack(side="right")
        counts = tk.Frame(stats, bg=WHITE)
        counts.pack(fill="x", pady=(1, 4))
        self._label(counts, variable=self.metrics["markers"], fg=MUTED, font=(FONT, 11)).pack(side="left")
        self._label(counts, variable=self.metrics["corners"], fg=MUTED, font=(FONT, 11)).pack(side="right")
        self._label(stats, f"JITTER / LAST {self.args.rolling_seconds:g} S / RMS", fg=MUTED,
                    font=(FONT, 10, "bold"), anchor="w").pack(fill="x", pady=(2, 3))
        jitters = tk.Frame(stats, bg=WHITE)
        jitters.pack(fill="x")
        for index, (caption, key) in enumerate((("Position", "position_jitter"),
                ("Rotation", "rotation_jitter"), ("Corners", "corner_jitter"))):
            jitters.columnconfigure(index, weight=1, uniform="jitter")
            cell = tk.Frame(jitters, bg=WHITE)
            cell.grid(row=0, column=index, sticky="ew")
            self._label(cell, caption, fg=MUTED, font=(FONT, 10), anchor="w").pack(fill="x")
            self._label(cell, variable=self.metrics[key], font=(MONO, 13, "bold"), anchor="w").pack(fill="x")

    def _build_table(self):
        tk, ttk = self.tk, self.ttk
        table_page = tk.Frame(self.notebook, bg=WHITE, padx=12, pady=8)
        self.notebook.add(table_page, text="Marker coordinates")
        note = self._label(table_page, variable=self.coordinate_note_var, fg=MUTED, font=(FONT, 11), anchor="w", justify="left", wraplength=1200)
        note.pack(fill="x", pady=(0, 6))
        table_page.bind("<Configure>", lambda event: note.configure(wraplength=max(100, event.width - 24)))
        self._label(table_page, variable=self.ids_var, fg=MUTED, font=(FONT, 10), anchor="w", justify="left",
                    wraplength=1200).pack(side="bottom", fill="x", pady=(6, 0))
        container = tk.Frame(table_page, bg=WHITE)
        container.pack(fill="both", expand=True)
        columns = ("id", "match", "u", "v", "x", "y", "z")
        self.coordinate_table = ttk.Treeview(container, columns=columns, show="headings", height=4)
        for key, caption, width in zip(columns, ("Marker ID", "Board match", "u / px", "v / px", "X / mm", "Y / mm", "Z / mm"),
                                        (90, 130, 130, 130, 160, 160, 160)):
            self.coordinate_table.heading(key, text=caption, anchor="w" if key == "match" else "e")
            self.coordinate_table.column(key, width=width, minwidth=70, anchor="w" if key == "match" else "e")
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.coordinate_table.yview)
        self.coordinate_table.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.coordinate_table.pack(side="left", fill="both", expand=True)
        self.coordinate_table.bind("<<TreeviewSelect>>", self._on_table_select)
        self.coordinate_table.tag_configure("foreign", foreground=AMBER)
        self.coordinate_table.tag_configure("rejected", foreground=RED)

    def _build_event_log(self):
        tk, ttk = self.tk, self.ttk
        page = tk.Frame(self.notebook, bg=WHITE, padx=12, pady=8)
        self.notebook.add(page, text="Event log")
        container = tk.Frame(page, bg=WHITE)
        container.pack(fill="both", expand=True)
        self.event_table = ttk.Treeview(container, columns=("time", "level", "message"), show="headings", height=4)
        for key, caption, width, stretch in (("time", "Time", 160, False), ("level", "Level", 90, False),
                                             ("message", "Message", 900, True)):
            self.event_table.heading(key, text=caption, anchor="w")
            self.event_table.column(key, width=width, minwidth=60, anchor="w", stretch=stretch)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.event_table.yview)
        self.event_table.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.event_table.pack(side="left", fill="both", expand=True)
        for level, color in LEVEL_COLORS.items():
            self.event_table.tag_configure(logging.getLevelName(level), foreground=color)
        self._event_ids = []

    def _log(self, level, message):
        """Write to the log file and the in-window event log (newest first, bounded)."""
        log.log(level, message)
        if self.destroyed:
            return
        name = logging.getLevelName(level)
        item = self.event_table.insert("", 0, values=(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name.title(), message),
                                       tags=(name,))
        self._event_ids.append(item)
        if len(self._event_ids) > EVENT_ROWS:
            self.event_table.delete(self._event_ids.pop(0))

    def _fullscreen(self, enabled):
        self.root.attributes("-fullscreen", bool(enabled))

    def _set_status(self, text, tone):
        self.status_var.set(text)
        self.status_badge.configure(bg=TONES[tone])

    def browse(self):
        from tkinter import filedialog
        selected = filedialog.askdirectory(parent=self.root, title="Choose where to save recordings",
                                           initialdir=str(Path(self.output_var.get() or ".").expanduser().resolve()))
        if selected:
            self.output_var.set(selected)
            self._next_disk_check = 0.

    def _clear_measurements(self):
        self.last_image = self.last_report = None
        self.last_frame_at = None
        self.copy_button.configure(state="disabled")
        for variable in self.metrics.values():
            variable.set("--")
        self.coordinate_table.delete(*self.coordinate_table.get_children())
        self.ids_var.set("Marker IDs will appear when the board is detected.")
        self.pose_state_var.set("No pose available")
        self.pose_state_label.configure(fg=MUTED)
        self.quality_var.set("Place the selected board in view.")
        self.performance_var.set("No frames received")
        self._render_image()

    def connect(self):
        if self.closing or self.disconnecting:
            return
        if self.worker is not None and self.worker.is_alive():
            self.disconnecting = True
            self._set_status("Disconnecting", "idle")
            for button in (self.connect_button, self.record_button, self.stop_button):
                button.configure(state="disabled")
            self.worker.disconnect()
            return
        args = copy.copy(self.args)
        args.board = self.board_var.get()
        args.camera_model = self.camera_model_var.get()
        self.worker = self.worker_factory(args)
        self.connected = self.reconnecting = self.stall_reported = False
        self._clear_measurements()
        for chart in self.charts:
            chart.set_points([])
        self._set_status("Connecting", "idle")
        self.recognition_var.set("Waiting for camera")
        self.recognition_label.configure(fg=MUTED)
        self.record_var.set("Preview only")
        self.board_box.configure(state="disabled")
        self.camera_model_box.configure(state="disabled")
        # Connecting can retry indefinitely; the button stays available to cancel it.
        self.connect_button.configure(text="Disconnect", state="normal")
        self._log(logging.INFO, f"Connecting to camera model {args.camera_model!r} for board {args.board!r}")
        if getattr(self.args, "auto_record", False):
            self._request_recording()
        self.worker.start()

    def _request_recording(self):
        parent = self.output_var.get().strip()
        if not parent:
            self.saved_var.set("Choose a folder before recording.")
            return
        self.pending_record = True
        self.record_button.configure(state="disabled")
        self._record_options("disabled")
        self.record_var.set("Starting recording...")
        self.worker.start_recording(recording_path(parent), self.save_frames_var.get())

    def start_recording(self):
        if not self.connected or self.recording or self.pending_record or self.closing or self.disconnecting:
            return
        self._request_recording()

    def stop_recording(self):
        if self.recording and self.worker is not None:
            self.stop_button.configure(state="disabled")
            self.record_var.set("Saving recording...")
            self.worker.stop_recording()

    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def _record_options(self, state):
        for widget in (self.output_entry, self.browse_button, self.save_check):
            widget.configure(state=state)

    def _recording_ended(self):
        self.recording = self.pending_record = False
        self.record_part = None
        self.stop_button.configure(state="disabled")
        self.record_button.configure(state="normal" if self.connected and not self.disconnecting else "disabled")
        self._record_options("normal")
        self.rec_pill.pack_forget()

    def _show_rec_pill(self, text, tone):
        self.rec_pill_var.set(text)
        self.rec_pill.configure(bg=TONES[tone])
        if not self.rec_pill.winfo_manager():
            self.rec_pill.pack(side="right", after=self.status_badge)

    def _event(self, kind, payload):
        described = describe_event(kind, payload)
        if described is not None:
            self._log(*described)
        if kind == "ready":
            if self.disconnecting:
                return
            self.connected, self.reconnecting = True, False
            self.retry_at = self.lost_error = None
            self._set_status("Camera live", "live")
            self.device_var.set(payload["camera"].get("stream_info", "Color camera"))
            self.connect_button.configure(text="Disconnect", state="normal")
            self.record_button.configure(state="disabled" if self.recording or self.pending_record else "normal")
            self.stop_button.configure(state="normal" if self.recording else "disabled")
            if self.recording:
                self.record_var.set("Resuming recording...")
            if payload.get("reconnected"):
                self.saved_var.set("Camera reconnected." + (" Recording continues in a new part." if self.recording else ""))
            self._render_image()
        elif kind == "record_started":
            self.recording, self.pending_record = True, False
            self.last_session = payload["path"]
            self.record_part = payload["part"]
            if not self.disconnecting:
                self.stop_button.configure(state="normal")
            self.record_button.configure(state="disabled")
            self._record_options("disabled")
            self.record_var.set("Recording" if payload["part"] == 1 else f"Recording part {payload['part']}")
            self._show_rec_pill("REC", "rec")
            if payload["part"] == 1:
                self.saved_var.set(f"Recording coordinates and jitter to {payload['path']}")
        elif kind == "record_stopped":
            if payload["path"] is not None:
                self.last_session = payload["path"]
            result = payload["summary"]
            if not payload["final"]:
                # A part closed by rotation or camera loss; the series continues.
                if payload["reason"] == "reconnect":
                    self.record_var.set(f"Paused: part {payload['part']} saved, waiting for camera")
                    self._show_rec_pill("REC PAUSED", "warn")
                return
            self._recording_ended()
            if result is None:
                self.record_var.set("Recording stopped")
                self.saved_var.set("Recording stopped." + (f" Last part: {payload['path']}" if payload["path"] else ""))
                return
            self.record_var.set(f"Saved {result['frames_recorded']:,} frames" +
                                (f" in part {payload['part']}" if payload["part"] > 1 else ""))
            rate = result["rates"]["board_recognized"]
            prefix = "Partial recording saved" if result["status"] != "complete" else "Recording saved"
            message = f"{prefix}: {payload['path']}"
            message += f" / Board detected in {rate:.0%} of frames" if rate is not None else " / No frames captured"
            if result["error"]:
                message += f" / {result['error']}"
            self.saved_var.set(message)
        elif kind == "camera_lost":
            self.connected, self.reconnecting = False, True
            self.lost_error, self.lost_attempt = payload["error"], payload["attempt"]
            self.retry_at = time.monotonic() + payload["retry_in"]
            self._set_status("Reconnecting", "warn")
            self.device_var.set("Camera unavailable")
            self.recognition_var.set("Camera unavailable")
            self.recognition_label.configure(fg=AMBER)
            self.record_button.configure(state="disabled")
            self.saved_var.set(f"Camera unavailable: {payload['error']}. Reconnecting automatically"
                               + ("; recording resumes in a new part." if self.recording or self.pending_record else "."))
            self._clear_measurements()
        elif kind == "warning":
            self.saved_var.set(f"Warning: {payload}")
        elif kind in ("error", "record_error"):
            self.saved_var.set(payload)
            if kind == "record_error":
                self._recording_ended()
                self.record_var.set("Recording stopped")
            else:
                self._set_status("Camera error", "error")
        elif kind == "disconnected":
            self.connected = self.recording = self.pending_record = self.disconnecting = self.reconnecting = False
            self.retry_at = None
            if self.status_var.get() != "Camera error":
                self._set_status("Disconnected", "idle")
            self.device_var.set("No camera connected")
            self.recognition_var.set("Camera disconnected")
            self.recognition_label.configure(fg=MUTED)
            self._clear_measurements()
            self._recording_ended()
            self.board_box.configure(state="readonly")
            self.camera_model_box.configure(state="readonly")
            self.connect_button.configure(text="Connect camera", state="normal")
            self.record_button.configure(state="disabled")

    def _poll(self):
        try:
            self._poll_once()
        except Exception:
            # Rescheduling below must survive any UI error, or signals and saving stop.
            log.exception("UI update failed")
        if not self.destroyed:
            self.root.after(50, self._poll)

    def _poll_once(self):
        # This frequent Python callback also lets pending OS signals run while Tcl
        # owns the event loop. The signal handler itself never touches Tk.
        if self.shutdown is not None and self.shutdown.requested.is_set() and not self.closing:
            name = signal.Signals(self.shutdown.signal_number).name
            self._log(logging.INFO, f"{name} received; saving and closing")
            self.close(KeyboardInterrupt(f"Stopped by {name}"))
        if self.worker is not None:
            while True:
                try:
                    kind, payload = self.worker.events.get_nowait()
                except Empty:
                    break
                if self.closing:
                    described = describe_event(kind, payload)
                    if described is not None:
                        log.log(*described)
                else:
                    self._event(kind, payload)
            if not self.closing and self.connected and not self.disconnecting:
                try:
                    preview = self.worker.previews.get_nowait()
                except Empty:
                    pass
                else:
                    self._show_preview(preview)
        if self.closing:
            self._poll_closing()
            return
        now = time.monotonic()
        self._update_health(now)
        self._update_overlay(now)
        self.watchdog.tick(self.worker)

    def _poll_closing(self):
        elapsed = time.monotonic() - self.close_started
        if self.worker is None or not self.worker.is_alive():
            self._destroy()
            return
        if elapsed > 4:
            self._set_status("Waiting for camera", "warn")
            self.saved_var.set("Closing the camera and saving the recording. "
                               "If the device is unresponsive, run ./kill.sh from a terminal.")
        if _give_up_waiting(self.worker, elapsed):
            log.error("Camera did not release within %.0f s; exiting without waiting for it", elapsed)
            self.abandoned = True
            self._destroy()

    def _destroy(self):
        self.destroyed = True
        self.root.destroy()

    def _update_health(self, now):
        stalled = self.watchdog.worker_stalled(self.worker, now)
        if stalled is not None and not self.stall_reported:
            self.stall_reported = True
            self._set_status("Camera stalled", "error")
            self._log(logging.ERROR, f"Capture thread has made no progress for {stalled:.0f} s; the USB device may be hung. "
                                     "Quit and restart, or run ./kill.sh.")
        elif stalled is None and self.stall_reported:
            self.stall_reported = False
            self._log(logging.INFO, "Capture thread is responding again")
            if self.connected:
                self._set_status("Camera live", "live")
        if now >= self._next_disk_check:
            self._next_disk_check = now + 5.
            free = _free_bytes(self.output_var.get().strip() or ".")
            minimum = getattr(self.args, "min_free_gb", 0.) * 1e9
            self.disk_var.set(f"Free {_gigabytes(free)}")
            self.disk_label.configure(fg=RED if free is not None and free < minimum
                                      else AMBER if free is not None and free < 5 * minimum else MUTED)
        if self.worker is not None:
            self.health = self.worker.health_snapshot(now)
        health = self.health
        parts = [f"Uptime {_duration(now - self.started)}"]
        if health:
            parts += [f"Frames {health['frames']:,}", f"Camera frame drops {health['camera_frame_gaps']:,}",
                      f"Dropouts {health['camera_losses']}", f"Processing errors {health['processing_errors']}"]
        if self.log_path:
            parts.append(f"Log {self.log_path}")
        self.health_var.set("   /   ".join(parts))
        if self.recording and self.connected:
            text = f"REC {_duration(self.record_seconds)}"
            self._show_rec_pill(text + (f" / part {self.record_part}" if self.record_part and self.record_part > 1 else ""), "rec")

    def _update_overlay(self, now):
        text = None
        if self.connected and self.last_frame_at is not None and now - self.last_frame_at > STALE_SECONDS:
            text = f"No new frames for {now - self.last_frame_at:.0f} s"
        if text != self._overlay_text:
            self._overlay_text = text
            self._draw_overlay()
        if self.reconnecting and self.retry_at is not None:
            remaining = max(0, math.ceil(self.retry_at - now))
            if remaining != self._last_countdown:
                self._last_countdown = remaining
                self._render_image()

    def _draw_overlay(self):
        canvas = self.preview
        canvas.delete("overlay")
        if not self._overlay_text:
            return
        width = canvas.winfo_width()
        item = canvas.create_text(width / 2, 22, text=self._overlay_text, fill=WHITE, font=(FONT, 11, "bold"), tags="overlay")
        x0, y0, x1, y1 = canvas.bbox(item)
        canvas.tag_lower(canvas.create_rectangle(x0 - 12, y0 - 6, x1 + 12, y1 + 6, fill=AMBER, outline="", tags="overlay"), item)

    def _show_preview(self, preview):
        report = preview["report"]
        self.last_report = report
        self.last_image = preview["image"]
        self.last_frame_at = time.monotonic()
        self.record_seconds = preview.get("recording_seconds", 0.)
        self._render_image()
        recognized = report["board_recognized"]
        self.recognition_var.set("Board detected" if recognized else "Board not detected")
        self.recognition_label.configure(fg=GREEN if recognized else AMBER)
        self.metrics["markers"].set(f"{len(report['matched_marker_ids'])} / {len(report['expected_marker_ids'])} markers")
        self.metrics["corners"].set(f"{report['detected_charuco_corner_count']} / {report['expected_charuco_corner_count']} corners")
        self.ids_var.set("\n".join(f"{caption}: {', '.join(map(str, report[key])) or 'none'}" for caption, key in
                                  (("Matched IDs", "matched_marker_ids"), ("Missing IDs", "missing_marker_ids"), ("Other IDs", "foreign_marker_ids"))))
        pose = report["pose"]
        self._show_position(report)
        self.metrics["fit"].set(f"Fit {_number(pose['reprojection_rmse_px'], 'px', 2)}" if pose else "Fit -- px")
        point = report.get("live_jitter") or {}
        for key, source, unit in (("position_jitter", "position_jitter_mm", "mm"), ("rotation_jitter", "rotation_jitter_deg", "deg"), ("corner_jitter", "corner_jitter_px", "px")):
            self.metrics[key].set(_number(point.get(source), unit))
        passed = report["pose_quality_passed"]
        self.pose_state_var.set(("Raw pose / Quality passed" if passed else "Raw pose / Check quality") if pose else "No pose available")
        self.pose_state_label.configure(fg=(GREEN if passed else AMBER) if pose else MUTED)
        if not recognized:
            self.quality_var.set("Place the selected board in view. Missing observations leave gaps in the plots.")
        elif not pose:
            self.quality_var.set("Markers found. Show more of the board to estimate its position.")
        elif passed:
            self.quality_var.set("Raw, unfiltered pose. Image fit is not physical accuracy.")
        else:
            reasons = report["pose_quality_reasons"]
            message = QUALITY_TEXT.get(reasons[0], reasons[0].replace("_", " ").capitalize()) if reasons else "Review the pose quality."
            if len(reasons) > 1:
                message += f" (+{len(reasons) - 1} more checks failed)"
            self.quality_var.set(message)
        self.performance_var.set(f"{_number(report.get('live_fps'), 'fps', 1)} / Detection {report['processing_ms']:.1f} ms"
                                 f" / Frame {report.get('camera_frame_number', '--')}")
        if self.recording and preview.get("recording"):
            text = f"REC {_duration(self.record_seconds)} / {preview.get('recorded_frames', 0):,} frames"
            self.record_var.set(text + (f" / part {self.record_part}" if self.record_part and self.record_part > 1 else ""))
        for chart in self.charts:
            chart.set_points(preview["points"])
        self._show_coordinates(report)

    def _selected_position(self, report):
        """(marker record or None, position in mm or None) for the selected marker."""
        marker = next((m for m in report.get("marker_coordinates", []) if m["id"] == self.selected_marker), None)
        return marker, marker["position_mm"] if marker else None

    def _show_position(self, report):
        if self.selected_marker is None:
            position = report["pose"]["position_mm"] if report["pose"] else None
            self.position_title_var.set("BOARD ORIGIN")
            self.position_note_var.set("Outer top-left pattern corner. Click a marker to show its centre.")
        else:
            marker, position = self._selected_position(report)
            self.position_title_var.set(f"MARKER #{self.selected_marker}")
            if marker is None:
                note = "Not visible. Click empty space in the view to show the board origin."
            elif position is None:
                note = ("Duplicate ID: position is ambiguous." if marker["duplicate"] else
                        "Not on the selected board: no position." if not marker["matched"] else
                        "No board pose yet: show more of the board.")
            else:
                note = "Marker centre on the board." + ("" if report["pose_quality_passed"] else
                                                         " Pose did not pass quality checks.")
            self.position_note_var.set(note)
        for index, (axis, _) in enumerate(AXES):
            self.metrics[axis.lower()].set(f"{position[index]:+.2f}" if position else "--")
        self.copy_button.configure(state="normal" if position else "disabled")

    def _select_marker(self, marker_id):
        self.selected_marker = marker_id
        if self.last_report is not None:
            self._show_position(self.last_report)
            self._sync_table_selection()
        else:
            self.position_title_var.set(f"MARKER #{marker_id}" if marker_id is not None else "BOARD ORIGIN")
        self._render_image()

    def _on_preview_click(self, event):
        report, display = self.last_report, self._display
        if report is None or display is None:
            return
        x0, y0, scale = display
        best, distance = None, 30.  # Display pixels.
        for marker in report.get("marker_coordinates", []):
            u, v = (value * scale for value in marker["pixel_center"])
            d = math.hypot(event.x - x0 - u, event.y - y0 - v)
            if d < distance:
                best, distance = marker["id"], d
        self._select_marker(None if best is None or best == self.selected_marker else best)

    def _on_table_select(self, event):
        selection = self.coordinate_table.selection()
        if selection:
            marker_id = int(str(selection[0]).split(":")[0])
            if marker_id != self.selected_marker:
                self._select_marker(marker_id)

    def _sync_table_selection(self):
        wanted = [item for item in self.coordinate_table.get_children()
                  if self.selected_marker is not None and str(item).split(":")[0] == str(self.selected_marker)][:1]
        if list(self.coordinate_table.selection()) != wanted:
            self.coordinate_table.selection_set(wanted)
            if wanted:
                self.coordinate_table.see(wanted[0])

    def _show_coordinates(self, report):
        existing = set(self.coordinate_table.get_children())
        counts = {}
        for index, marker in enumerate(report.get("marker_coordinates", [])):
            occurrence = counts.get(marker["id"], 0)
            counts[marker["id"]] = occurrence + 1
            item = f"{marker['id']}:{occurrence}"
            state = "Duplicate ID" if marker["duplicate"] else "Matched" if marker["matched"] else "Other board"
            position = marker["position_mm"] or [None] * 3
            tag = "foreign" if not marker["matched"] else "rejected" if not report["pose_quality_passed"] else ""
            values = (marker["id"], state,
                      *(_number(v, digits=2) for v in marker["pixel_center"]),
                      *(_number(v, digits=2) for v in position))
            if item in existing:
                existing.remove(item)
                self.coordinate_table.item(item, values=values, tags=(tag,))
                self.coordinate_table.move(item, "", index)
            else:
                self.coordinate_table.insert("", index, iid=item, values=values, tags=(tag,))
        if existing:
            self.coordinate_table.delete(*existing)
        self._sync_table_selection()
        suffix = " / Current raw pose did not pass quality checks." if report["pose"] and not report["pose_quality_passed"] else ""
        self.coordinate_note_var.set("u/v: mean of detected corners. X/Y/Z: board-model centres in the camera frame, inferred from the raw pose." + suffix)

    def _redraw_charts(self):
        for chart in self.charts:
            if chart.canvas.winfo_viewable():
                chart.draw()

    def copy_coordinates(self):
        report = self.last_report
        if report is None:
            return
        if self.selected_marker is None:
            values, subject = (report["pose"]["position_mm"] if report["pose"] else None), "Board origin"
        else:
            values, subject = self._selected_position(report)[1], f"Marker {self.selected_marker} centre"
        if values is None:
            return
        text = f"{subject} / camera frame / mm\n" + "\n".join(f"{axis}={value:+.2f}" for (axis, _), value in zip(AXES, values))
        text += f"\nCamera axes: X right, Y down, Z forward\nQuality passed: {report['pose_quality_passed']}"
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.saved_var.set(f"Copied {subject.lower()} coordinates in mm, with camera-frame and quality labels.")

    def _preview_message(self):
        if self.reconnecting:
            remaining = max(0, math.ceil(self.retry_at - time.monotonic())) if self.retry_at else 0
            retry = f"Retrying in {remaining} s" if remaining else "Retrying now"
            return f"Camera unavailable\n{self.lost_error}\n\n{retry} (attempt {self.lost_attempt})"
        if self.worker is not None and self.worker.is_alive():
            return "Connecting to the camera..."
        return "Connect a camera to begin"

    def _render_image(self):
        from PIL import Image, ImageTk
        width, height = self.preview.winfo_width(), self.preview.winfo_height()
        if self.last_image is None:
            self.preview.delete("all")
            self.image_item = None
            self._display = None
            self.preview.create_text(width / 2, height / 2, text=self._preview_message(), fill="#a4beca",
                                     font=(FONT, 13), justify="center", width=max(200, width - 80))
            self._draw_overlay()
            return
        image = Image.fromarray(self.last_image[:, :, ::-1])
        full_width = image.width
        image.thumbnail((max(1, width), max(1, height)), Image.Resampling.BILINEAR)
        scale = image.width / full_width
        self._display = ((width - image.width) / 2, (height - image.height) / 2, scale)
        report = self.last_report
        if report is not None and self.marker_labels_var.get() and report.get("marker_coordinates"):
            image = annotate_markers(image, report["marker_coordinates"], scale, self.selected_marker,
                                     report["pose_quality_passed"])
        # Reuse one Tk image while the size is unchanged; avoids per-frame image churn.
        if self.photo is not None and (self.photo.width(), self.photo.height()) == image.size:
            self.photo.paste(image)
        else:
            self.photo = ImageTk.PhotoImage(image, master=self.root)
        if self.image_item is None:
            self.preview.delete("all")
            self.image_item = self.preview.create_image(width / 2, height / 2, image=self.photo, anchor="center")
            self._draw_overlay()
        else:
            self.preview.itemconfigure(self.image_item, image=self.photo)
            self.preview.coords(self.image_item, width / 2, height / 2)

    def close(self, error=None):
        if self.closing:
            return
        self.closing = True
        self.close_started = time.monotonic()
        self.watchdog.stopping()
        self._set_status("Saving and closing", "idle")
        for button in (self.connect_button, self.record_button, self.stop_button, self.copy_button):
            button.configure(state="disabled")
        if self.worker is not None:
            self.worker.disconnect(error)


def run_gui(args):
    try:
        import tkinter as tk
        from PIL import ImageTk  # noqa: F401
    except ImportError as exc:
        raise ValueError("Desktop UI needs Tkinter and Pillow. Install Python's Tk support and pip install -e '.[gui]'.") from exc
    _, record, manifest = _load_board(args.manifest, args.board, args.generator_path)
    args.board = record["name"]
    geometry = record["geometry"]
    width = geometry["squares_x"] * geometry["square_length_mm"]
    height = geometry["squares_y"] * geometry["square_length_mm"]
    selection = f"Board {args.board!r} ({width:g} x {height:g} mm); manifest: {args.manifest}"
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise ValueError(f"Cannot open desktop UI: {exc}. Use a desktop display or the service command.") from exc
    log_path = configure_logging(default_log_file(args))
    log.info(selection)
    with SignalStop() as shutdown:
        app = MonitorApp(root, args, [board["name"] for board in manifest["boards"]], shutdown=shutdown, log_path=log_path)
        try:
            root.mainloop()
        except KeyboardInterrupt:
            app.close(KeyboardInterrupt("Interrupted"))
            root.mainloop()
    code = 128 + shutdown.signal_number if shutdown.signal_number is not None else 0
    if app.worker is not None and app.worker.is_alive():
        # The capture thread is stuck in the USB stack after its recording was
        # saved; a normal interpreter exit would wait for it forever.
        logging.shutdown()
        os._exit(code)
    return code
