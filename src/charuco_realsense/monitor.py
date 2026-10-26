"""Desktop monitor data model and camera worker; no GUI dependencies."""
from collections import deque
import copy
import csv
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time

import numpy as np

from .calibration import load_calibration, opencv_calibration
from .cli import (_analyze, _draw, _frame_fields, _load_board, _metadata,
                  _open_camera, _pose_options, _settle_camera)
from .detect import CharucoDetector
from .metrics import _mean_rotation, load_reference_pose, pose_error, rotation_angle_deg
from .recording import SessionRecorder


class LiveHistory:
    """Raw pose displacement and sliding-window jitter, with explicit gaps."""

    def __init__(self, rolling_seconds=2., window_seconds=30.):
        if not np.isfinite([rolling_seconds, window_seconds]).all() or min(rolling_seconds, window_seconds) <= 0:
            raise ValueError("Monitor windows must be finite and positive")
        self.rolling_seconds = rolling_seconds
        self.window_seconds = window_seconds
        self.recent = deque(maxlen=5000)
        self.points = deque(maxlen=5000)
        self.baseline = None
        self.started = None

    def add(self, report, now):
        if self.started is None:
            self.started = now
        self.recent.append((now, report["pose"], report["charuco_corners"]))
        while self.recent and self.recent[0][0] < now - self.rolling_seconds:
            self.recent.popleft()
        current = report["pose"]
        delta = None
        position_jitter = rotation_jitter = corner_jitter = None
        if current is not None:
            position = np.asarray(current["tvec_m"], float) * 1000.
            if self.baseline is None:
                self.baseline = position.copy()
            delta = (position - self.baseline).tolist()
            poses = [pose for _, pose, _ in self.recent if pose is not None]
            if len(poses) >= 2:
                positions = np.asarray([pose["tvec_m"] for pose in poses]) * 1000.
                position_jitter = float(np.sqrt(np.mean(np.sum((positions - positions.mean(axis=0)) ** 2, axis=1))))
                rotations = [pose["rotation_matrix"] for pose in poses]
                reference = _mean_rotation(rotations)
                rotation_jitter = float(np.sqrt(np.mean([rotation_angle_deg(r, reference) ** 2 for r in rotations])))
        if report["charuco_corners"]:
            corners = {}
            for _, _, observations in self.recent:
                ids = [p["id"] for p in observations]
                for p in observations:
                    if ids.count(p["id"]) == 1:
                        corners.setdefault(p["id"], []).append(p["pixel"])
            variances = [np.var(pixels, axis=0).sum() for pixels in corners.values() if len(pixels) >= 2]
            if variances:
                corner_jitter = float(np.sqrt(np.mean(variances)))
        point = {"elapsed_s": now - self.started, "delta_xyz_mm": delta,
                 "position_jitter_mm": position_jitter, "rotation_jitter_deg": rotation_jitter,
                 "corner_jitter_px": corner_jitter, "rolling_seconds": self.rolling_seconds}
        self.points.append(point)
        while self.points and self.points[0]["elapsed_s"] < point["elapsed_s"] - self.window_seconds:
            self.points.popleft()
        return point


CSV_FIELDS = ("elapsed_s", "delta_x_mm", "delta_y_mm", "delta_z_mm", "position_jitter_mm",
              "rotation_jitter_deg", "corner_jitter_px", "board_recognized", "pose_available",
              "pose_quality_passed", "matched_markers", "charuco_corners")


class CaptureWorker(threading.Thread):
    """Own the camera/recorder off the Tk thread; only preview snapshots may drop."""

    def __init__(self, args, camera_factory=None):
        super().__init__(name="charuco-camera")
        self.args = copy.copy(args)
        self.camera_factory = camera_factory or _open_camera
        self.events = Queue()
        self.previews = Queue(maxsize=1)
        self.commands = Queue()
        self.stop_requested = threading.Event()
        self.recorder = None
        self.csv_file = None
        self.csv_writer = None
        self.last_summary = None
        self.history = LiveHistory(args.rolling_seconds, args.window_seconds)
        self.started_at = None
        self.recorded_frames = 0
        self._last_publish = 0.

    def start_recording(self, path, save_frames=False):
        self.commands.put(("start", (str(path), save_frames)))

    def stop_recording(self):
        self.commands.put(("stop", None))

    def disconnect(self):
        self.stop_requested.set()

    def _finish_recording(self, error=None):
        if self.recorder is None:
            return
        recorder = self.recorder
        self.recorder = None
        try:
            if self.csv_file is not None:
                self.csv_file.close()
        finally:
            self.csv_file = self.csv_writer = None
            recorder.__exit__(type(error) if error else None, error, None)
        self.last_summary = recorder.summary
        self.events.put(("record_stopped", {"path": str(recorder.path), "summary": recorder.summary}))

    def _commands(self, metadata, camera):
        while True:
            try:
                command, payload = self.commands.get_nowait()
            except Empty:
                return
            if command == "stop":
                self._finish_recording()
            elif command == "start" and self.recorder is None:
                path, save_frames = payload
                try:
                    session_metadata = copy.deepcopy(metadata)
                    session_metadata["created_at_utc"] = datetime.now(timezone.utc).isoformat()
                    session_metadata["camera"] = camera.metadata
                    session_metadata["recording"]["save_frames"] = save_frames
                    session_metadata["monitor"] = {"rolling_seconds": self.args.rolling_seconds,
                                                    "window_seconds": self.args.window_seconds}
                    self.recorder = SessionRecorder(path, session_metadata, save_frames)
                    self.csv_file = (Path(path) / "jitter.csv").open("w", newline="", encoding="utf-8")
                    self.csv_writer = csv.writer(self.csv_file)
                    self.csv_writer.writerow(CSV_FIELDS)
                    self.csv_file.flush()
                except (OSError, ValueError) as exc:
                    self._finish_recording(exc)
                    self.events.put(("record_error", str(exc)))
                    continue
                self.history = LiveHistory(self.args.rolling_seconds, self.args.window_seconds)
                self.started_at = time.monotonic()
                self.history.started = self.started_at
                self.recorded_frames = 0
                self.events.put(("record_started", path))

    def run(self):
        error = None
        try:
            args = self.args
            board, record, manifest = _load_board(args.manifest, args.board, args.generator_path)
            detector = CharucoDetector(board, record["geometry"]["border_bits"])
            options = _pose_options(args)
            calibration = load_calibration(args.calibration) if args.calibration else None
            reference = load_reference_pose(args.reference_pose) if args.reference_pose else None
            with self.camera_factory(args) as camera:
                _settle_camera(camera, args)
                if self.stop_requested.is_set():
                    return
                intrinsics = calibration if calibration is not None else camera.intrinsics
                opencv_calibration(intrinsics, (camera.intrinsics["width"], camera.intrinsics["height"]))
                metadata = _metadata(args, camera, record, manifest, detector, options, intrinsics, reference)
                self.events.put(("ready", {"camera": metadata["camera"], "board": record}))
                connected_at = time.monotonic()
                self.history.started = connected_at
                index = 0
                while not self.stop_requested.is_set():
                    self._commands(metadata, camera)
                    frame = camera.read()
                    now = time.monotonic()
                    detection, report = _analyze(board, record, detector, frame.bgr, intrinsics, True, options)
                    report.update(_frame_fields(frame, self.recorded_frames if self.recorder else index,
                                                now, self.started_at if self.recorder else connected_at))
                    point = self.history.add(report, now)
                    report["live_jitter"] = point
                    if reference is not None and report["pose"] is not None:
                        report["reference_error"] = pose_error(report["pose"], reference)
                    if self.recorder is not None:
                        self.recorder.add(report, frame.bgr)
                        self.csv_writer.writerow([
                            point["elapsed_s"], *(point["delta_xyz_mm"] or [None] * 3),
                            point["position_jitter_mm"], point["rotation_jitter_deg"], point["corner_jitter_px"],
                            report["board_recognized"], report["pose_available"], report["pose_quality_passed"],
                            len(report["matched_marker_ids"]), report["detected_charuco_corner_count"]])
                        self.csv_file.flush()
                        self.recorded_frames += 1
                    # Redraw charts/preview at up to 10 Hz without throttling recording.
                    if now - self._last_publish >= .1:
                        preview = {"image": _draw(frame.bgr.copy(), detection), "report": report,
                                   "points": list(self.history.points), "recorded_frames": self.recorded_frames,
                                   "recording": self.recorder is not None,
                                   "recording_seconds": now - self.started_at if self.recorder else 0.}
                        try:
                            self.previews.get_nowait()
                        except Empty:
                            pass
                        try:
                            self.previews.put_nowait(preview)
                        except Full:
                            pass
                        self._last_publish = now
                    index += 1
        except Exception as exc:
            error = exc
            self.events.put(("error", str(exc)))
        finally:
            try:
                self._finish_recording(error)
            except Exception as exc:
                self.events.put(("error", f"Could not finalize recording: {exc}"))
            self.events.put(("disconnected", None))
