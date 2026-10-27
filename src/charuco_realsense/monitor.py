"""Desktop monitor data model and camera worker; no GUI dependencies."""
from collections import Counter, deque
import copy
import csv
from datetime import datetime, timezone
import logging
from pathlib import Path
from queue import Empty, Full, Queue
import shutil
import threading
import time

import numpy as np

from .calibration import load_calibration, opencv_calibration
from .camera import CameraCancelled, CameraError, reset_device
from .cli import (_analyze, _draw, _frame_fields, _load_board, _metadata,
                  _open_camera, _pose_options, _settle_camera)
from .detect import CharucoDetector
from .coordinates import COORDINATE_FRAME, MarkerCoordinates
from .metrics import _mean_rotation, load_reference_pose, pose_error
from .recording import SessionRecorder

log = logging.getLogger("charuco_realsense")


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
                rotations = np.asarray([pose["rotation_matrix"] for pose in poses])
                reference = _mean_rotation(rotations)
                cosine = (np.einsum("nij,ij->n", rotations, reference) - 1.) / 2.
                angles = np.degrees(np.arccos(np.clip(cosine, -1., 1.)))
                rotation_jitter = float(np.sqrt(np.mean(angles ** 2)))
        if report["charuco_corners"]:
            corners = {}
            for _, _, observations in self.recent:
                ids = Counter(p["id"] for p in observations)
                for p in observations:
                    if ids[p["id"]] == 1:
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
              "pose_quality_passed", "matched_markers", "charuco_corners",
              "position_x_mm", "position_y_mm", "position_z_mm", "camera_timestamp_ms", "camera_frame_number")
MARKER_CSV_FIELDS = ("elapsed_s", "camera_frame_number", "id", "matched", "duplicate", "center_u_px",
                     "center_v_px", "position_x_mm", "position_y_mm", "position_z_mm", "pose_quality_passed")


def _free_bytes(path):
    """Free space on the filesystem that holds path, or its nearest existing parent."""
    path = Path(path).expanduser().absolute()
    for candidate in (path, *path.parents):
        try:
            return shutil.disk_usage(candidate).free
        except OSError:
            continue
    return None


class CaptureWorker(threading.Thread):
    """Own the camera/recorder off the UI thread; only preview snapshots may drop.

    For unattended use, a camera failure finalizes the current recording part,
    then the worker reconnects with exponential backoff and resumes recording in
    a new part directory. Long recordings also rotate into parts, which bounds
    per-part memory and file size. Each part is a complete recording directory.
    """

    def __init__(self, args, camera_factory=None, device_reset=None):
        super().__init__(name="charuco-camera")
        self.args = copy.copy(args)
        self.events = Queue()
        self.previews = Queue(maxsize=1)
        self.commands = Queue()
        self.stop_requested = threading.Event()
        self.camera_factory = camera_factory or (lambda args: _open_camera(args, self.stop_requested))
        self.device_reset = device_reset or (lambda args: reset_device(args.serial))
        self.shutdown_error = None
        # The requested recording series survives reconnects; the recorder is one part.
        self.request = None
        self.recorder = None
        self.csv_file = None
        self.csv_writer = None
        self.marker_file = None
        self.marker_writer = None
        self.last_summary = None
        self.history = LiveHistory(args.rolling_seconds, args.window_seconds)
        self.started_at = None
        self.connected_at = None
        self.recorded_frames = 0
        self.segment_frames = 0
        self.segment_started = None
        # Read by the UI/service thread to detect a capture loop stuck in the SDK.
        self.heartbeat = time.monotonic()
        self.health = {"started": self.heartbeat, "frames": 0, "camera_losses": 0, "reconnects": 0,
                       "processing_errors": 0, "camera_frame_gaps": 0, "last_error": None}
        self._live = False  # Whether the current camera session delivered "ready".
        self._last_frame_number = None
        self._next_disk_check = 0.
        self._last_publish = 0.
        self._frame_times = deque(maxlen=30)

    def start_recording(self, path, save_frames=False):
        self.commands.put(("start", (str(path), save_frames)))

    def stop_recording(self):
        self.commands.put(("stop", None))

    def disconnect(self, error=None):
        self.shutdown_error = error
        self.stop_requested.set()

    def _close_segment(self, error, reason, final):
        if self.recorder is None:
            return
        recorder = self.recorder
        self.recorder = None
        summary = None
        try:
            try:
                if self.csv_file is not None:
                    self.csv_file.close()
            finally:
                try:
                    if self.marker_file is not None:
                        self.marker_file.close()
                finally:
                    self.csv_file = self.csv_writer = self.marker_file = self.marker_writer = None
                    recorder.__exit__(type(error) if error else None, error, None)
            summary = recorder.summary
            self.last_summary = summary
        except (OSError, ValueError) as exc:
            # A full disk can prevent summary.json; frames.jsonl remains analysable.
            self.events.put(("warning", f"Could not finalize {recorder.path}: {exc}"))
        if self.request is not None:
            self.request["reason"] = reason
        self.events.put(("record_stopped", {"path": str(recorder.path), "summary": summary,
                                            "part": recorder.metadata.get("segment", {}).get("part", 1),
                                            "reason": reason, "final": final}))

    def _abort_recording(self, error):
        self.request = None
        self._close_segment(error, "error", final=True)
        self.events.put(("record_error", str(error)))

    def _commands(self):
        while True:
            try:
                command, payload = self.commands.get_nowait()
            except Empty:
                return
            if command == "stop":
                request, self.request = self.request, None
                if self.recorder is not None:
                    self._close_segment(None, "stopped", final=True)
                elif request is not None:
                    # Stopped while the camera was unavailable: no part is open.
                    self.events.put(("record_stopped", {"path": None, "summary": None, "part": request["part"],
                                                        "reason": "stopped", "final": True}))
            elif command == "start" and self.request is None:
                path, save_frames = payload
                self.request = {"path": Path(path), "save_frames": save_frames, "part": 0, "reason": "start"}

    def _maintain_recording(self, metadata, camera, now):
        request = self.request
        if request is None:
            return
        segment_seconds = self.args.segment_minutes * 60.
        if self.recorder is not None and segment_seconds and now - self.segment_started >= segment_seconds:
            self._close_segment(None, "rotation", final=False)
        if now >= self._next_disk_check or self.recorder is None:
            self._next_disk_check = now + 5.
            free = _free_bytes(request["path"].parent)
            minimum = self.args.min_free_gb * 1e9
            if free is not None and free < minimum:
                self._abort_recording(OSError(f"Low disk space: {free / 1e9:.1f} GB free; recording needs "
                                              f"at least {self.args.min_free_gb:g} GB (--min-free-gb)"))
                return
        if self.recorder is None:
            self._open_segment(metadata, camera, now)

    def _open_segment(self, metadata, camera, now):
        request = self.request
        part = request["part"] + 1
        base = request["path"]
        path = base if part == 1 else base.with_name(f"{base.name}-part{part:04d}")
        try:
            session_metadata = copy.deepcopy(metadata)
            session_metadata["created_at_utc"] = datetime.now(timezone.utc).isoformat()
            session_metadata["camera"] = camera.metadata
            session_metadata["recording"]["save_frames"] = request["save_frames"]
            session_metadata["monitor"] = {"rolling_seconds": self.args.rolling_seconds,
                                           "window_seconds": self.args.window_seconds}
            session_metadata["segment"] = {"series": base.name, "part": part, "reason": request["reason"],
                                           "segment_minutes": self.args.segment_minutes or None}
            session_metadata["coordinates"] = COORDINATE_FRAME
            self.recorder = SessionRecorder(path, session_metadata, request["save_frames"])
            self.csv_file = (path / "jitter.csv").open("w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(CSV_FIELDS)
            self.csv_file.flush()
            self.marker_file = (path / "markers.csv").open("w", newline="", encoding="utf-8")
            self.marker_writer = csv.writer(self.marker_file)
            self.marker_writer.writerow(MARKER_CSV_FIELDS)
            self.marker_file.flush()
        except (OSError, ValueError) as exc:
            self._abort_recording(exc)
            return
        request["part"] = part
        self.segment_started = now
        self.segment_frames = 0
        if part == 1:
            # A new series resets the live history and displacement baseline;
            # later parts continue them so elapsed time stays continuous.
            self.history = LiveHistory(self.args.rolling_seconds, self.args.window_seconds)
            self.started_at = now
            self.history.started = now
            self.recorded_frames = 0
        self.events.put(("record_started", {"path": str(path), "part": part, "reason": request["reason"]}))

    def _record(self, report, frame, point):
        self.recorder.add(report, frame.bgr)
        self.csv_writer.writerow([
            point["elapsed_s"], *(point["delta_xyz_mm"] or [None] * 3),
            point["position_jitter_mm"], point["rotation_jitter_deg"], point["corner_jitter_px"],
            report["board_recognized"], report["pose_available"], report["pose_quality_passed"],
            len(report["matched_marker_ids"]), report["detected_charuco_corner_count"],
            *((report["pose"]["position_mm"] if report["pose"] else None) or [None] * 3),
            frame.timestamp_ms, frame.frame_number])
        self.csv_file.flush()
        for marker in report["marker_coordinates"]:
            self.marker_writer.writerow([
                point["elapsed_s"], frame.frame_number, marker["id"], marker["matched"],
                marker["duplicate"], *marker["pixel_center"],
                *(marker["position_mm"] or [None] * 3), report["pose_quality_passed"]])
        self.marker_file.flush()
        self.segment_frames += 1
        self.recorded_frames += 1

    def _beat(self):
        self.heartbeat = time.monotonic()

    def health_snapshot(self, now=None):
        now = time.monotonic() if now is None else now
        return {**self.health, "uptime_s": now - self.health["started"],
                "part": self.request["part"] if self.request else None}

    def _wait(self, seconds):
        """Back off without blocking stop/record commands or the watchdog heartbeat."""
        deadline = time.monotonic() + seconds
        while not self.stop_requested.is_set():
            self.heartbeat = time.monotonic()
            self._commands()
            remaining = deadline - self.heartbeat
            if remaining <= 0:
                return
            self.stop_requested.wait(min(.25, remaining))

    def run(self):
        error = None
        try:
            args = self.args
            board, record, manifest = _load_board(args.manifest, args.board, args.generator_path)
            context = {"board": board, "record": record, "manifest": manifest,
                       "detector": CharucoDetector(board, record["geometry"]["border_bits"]),
                       "coordinates": MarkerCoordinates(board), "options": _pose_options(args),
                       "calibration": load_calibration(args.calibration) if args.calibration else None,
                       "reference": load_reference_pose(args.reference_pose) if args.reference_pose else None}
            attempt = 0
            while not self.stop_requested.is_set():
                self.heartbeat = time.monotonic()
                self._live = False
                try:
                    self._session(context)
                    break
                except CameraCancelled:
                    if self.stop_requested.is_set():
                        break
                    raise
                except CameraError as exc:
                    if not args.reconnect:
                        raise
                    attempt = 1 if self._live else attempt + 1
                    delay = min(args.reconnect_max_seconds, 2. ** min(attempt - 1, 10))
                    self.health["camera_losses"] += self._live
                    self.health["last_error"] = str(exc)
                    self.events.put(("camera_lost", {"error": str(exc), "attempt": attempt, "retry_in": delay,
                                                     "recording": self.request is not None}))
                    if args.hardware_reset_after and attempt % args.hardware_reset_after == 0:
                        self._reset_device()
                    self._wait(delay)
        except Exception as exc:
            error = exc
            self.health["last_error"] = str(exc)
            log.exception("Camera worker stopped")
            self.events.put(("error", str(exc)))
        finally:
            try:
                self._close_segment(error or self.shutdown_error, "stopped", final=True)
            except Exception as exc:
                self.events.put(("error", f"Could not finalize recording: {exc}"))
            self.request = None
            self.events.put(("disconnected", None))

    def _reset_device(self):
        try:
            if self.device_reset(self.args):
                self.events.put(("warning", "Sent a hardware reset to the camera; it re-enumerates in a few seconds"))
        except Exception as exc:
            self.events.put(("warning", f"Camera hardware reset failed: {exc}"))

    def _session(self, context):
        with self.camera_factory(self.args) as camera:
            try:
                self._stream(camera, **context)
            except BaseException as exc:
                # Finalize before the camera closes: stopping a pipeline on a lost
                # USB device can block, and the summary must not depend on it.
                if self.stop_requested.is_set() and isinstance(exc, CameraCancelled):
                    self._close_segment(self.shutdown_error, "stopped", final=True)
                else:
                    retry = (self.args.reconnect and isinstance(exc, CameraError)
                             and not isinstance(exc, CameraCancelled))
                    self._close_segment(exc, "reconnect" if retry else "error", final=not retry)
                raise
            self._close_segment(self.shutdown_error, "stopped", final=True)

    def _stream(self, camera, board, record, manifest, detector, coordinates, options, calibration, reference):
        args = self.args
        _settle_camera(camera, args, self.stop_requested, self._beat)
        if self.stop_requested.is_set():
            return
        intrinsics = calibration if calibration is not None else camera.intrinsics
        opencv_calibration(intrinsics, (camera.intrinsics["width"], camera.intrinsics["height"]))
        metadata = _metadata(args, camera, record, manifest, detector, options, intrinsics, reference)
        now = time.monotonic()
        reconnected = self.connected_at is not None
        if reconnected:
            self.health["reconnects"] += 1
        else:
            self.connected_at = now
            if self.history.started is None:
                self.history.started = now
        self._live = True
        self._last_frame_number = None
        self._frame_times.clear()
        self.events.put(("ready", {"camera": metadata["camera"], "board": record, "reconnected": reconnected}))
        index = 0
        failures = 0
        while not self.stop_requested.is_set():
            self.heartbeat = time.monotonic()
            self._commands()
            self._maintain_recording(metadata, camera, self.heartbeat)
            frame = camera.read()
            now = time.monotonic()
            self.heartbeat = now
            self.health["frames"] += 1
            if self._last_frame_number is not None and frame.frame_number > self._last_frame_number + 1:
                self.health["camera_frame_gaps"] += frame.frame_number - self._last_frame_number - 1
            self._last_frame_number = frame.frame_number
            try:
                detection, report = _analyze(board, record, detector, frame.bgr, intrinsics, True, options)
                report["marker_coordinates"] = coordinates.measure(detection, report["pose"])
            except Exception as exc:
                # One malformed frame must not end an unattended session.
                failures += 1
                self.health["processing_errors"] += 1
                count = self.health["processing_errors"]
                if count <= 5 or count % 1000 == 0:
                    log.exception("Frame %s could not be processed (%d so far)", frame.frame_number, count)
                    self.events.put(("warning", f"Frame processing error ({count} so far): {exc}"))
                if failures >= 30:
                    raise CameraError(f"{failures} consecutive frames could not be processed: {exc}") from exc
                continue
            failures = 0
            self._frame_times.append(now)
            report["live_fps"] = ((len(self._frame_times) - 1) / (now - self._frame_times[0])
                                  if len(self._frame_times) > 1 and now > self._frame_times[0] else None)
            report.update(_frame_fields(frame, self.segment_frames if self.recorder else index,
                                        now, self.started_at if self.recorder else self.connected_at))
            point = self.history.add(report, now)
            report["live_jitter"] = point
            if reference is not None and report["pose"] is not None:
                report["reference_error"] = pose_error(report["pose"], reference)
            if self.recorder is not None:
                try:
                    self._record(report, frame, point)
                except (OSError, ValueError) as exc:
                    # Storage failures stop recording; the live preview continues.
                    self._abort_recording(exc)
            # Redraw charts/preview at up to 10 Hz without throttling recording.
            if now - self._last_publish >= .1:
                preview = {"image": _draw(frame.bgr.copy(), detection, report["pose"], intrinsics, labels=False), "report": report,
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
