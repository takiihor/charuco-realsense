import csv
import copy
import json
from pathlib import Path
import time
from queue import Empty

import cv2
import numpy as np
import pytest

from charuco_realsense.camera import CameraError, Frame
from charuco_realsense.cli import parser
from charuco_realsense.monitor import CaptureWorker, LiveHistory


def observation(position=None, angle=0., pixel=None):
    pose = None
    if position is not None:
        rotation = cv2.Rodrigues(np.array([0., 0., np.radians(angle)]))[0]
        pose = {"tvec_m": list(position), "rotation_matrix": rotation.tolist()}
    return {"pose": pose, "charuco_corners": [{"id": 0, "pixel": pixel}] if pixel is not None else []}


def test_rolling_jitter_and_rotation_wraparound():
    history = LiveHistory(rolling_seconds=2., window_seconds=3.)
    first = history.add(observation((0, 0, .5), 179., [10, 20]), 0.)
    assert first["position_jitter_mm"] is None
    point = history.add(observation((.002, 0, .5), -179., [12, 24]), 1.)
    assert point["position_jitter_mm"] == pytest.approx(1.)
    assert point["rotation_jitter_deg"] == pytest.approx(1.)
    assert point["corner_jitter_px"] == pytest.approx(np.sqrt(5))
    np.testing.assert_allclose(point["delta_xyz_mm"], [2, 0, 0])
    gap = history.add(observation(), 1.5)
    assert gap["delta_xyz_mm"] is None
    assert gap["position_jitter_mm"] is None and gap["corner_jitter_px"] is None
    later = history.add(observation((.004, 0, .5), 179., [20, 20]), 4.)
    assert later["position_jitter_mm"] is None
    assert later["corner_jitter_px"] is None
    assert history.points[0]["elapsed_s"] == 1.


@pytest.fixture
def worker_setup(artifacts, tmp_path):
    directory, _ = artifacts
    image = cv2.imread(str(directory / "front_charuco.png"))
    blank = np.full_like(image, 255)
    height, width = image.shape[:2]
    args = parser().parse_args(["gui", "--board", "front", "--manifest", str(directory / "board_manifest.json"),
                               "--warmup-seconds", "0", "--output-dir", str(tmp_path)])

    class Camera:
        def __init__(self):
            self.index = 0
            self.closed = False
            self.callback = None
            self.intrinsics = {"fx": float(width), "fy": float(width), "ppx": width / 2, "ppy": height / 2,
                               "width": width, "height": height, "model": "none", "coeffs": [0.] * 5}
            self.metadata = {"intrinsics": self.intrinsics, "stream_info": "Test camera"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def read(self):
            time.sleep(.01)
            self.index += 1
            if self.callback:
                self.callback(self.index)
            return Frame(blank if self.index % 3 == 0 else image, self.intrinsics,
                         self.index * 100., self.index, "hardware_clock")

    camera = Camera()
    worker = CaptureWorker(args, camera_factory=lambda args: camera)
    return worker, camera, args


def events(worker):
    values = []
    while True:
        try:
            values.append(worker.events.get_nowait())
        except Empty:
            return values


def test_worker_hardware_recovery_uses_selected_camera_model(worker_setup, monkeypatch):
    from charuco_realsense import monitor
    _, _, args = worker_setup
    args.camera_model = "D415"
    calls = []
    monkeypatch.setattr(monitor, "reset_device", lambda serial, model: calls.append((serial, model)) or True)
    worker = CaptureWorker(args)
    worker._reset_device()
    assert calls == [(None, "D415")]


def test_worker_records_all_frames_while_preview_drops_old_frames(worker_setup, tmp_path):
    worker, camera, _ = worker_setup
    camera.callback = lambda index: worker.disconnect() if index == 8 else None
    directory = tmp_path / "run"
    worker.start_recording(directory)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive() and camera.closed
    assert worker.previews.qsize() == 1
    frames = [json.loads(line) for line in (directory / "frames.jsonl").read_text().splitlines()]
    assert len(frames) == 8
    assert frames[0]["frame_index"] == 0 and frames[-1]["frame_index"] == 7
    assert frames[2]["pose"] is None
    assert frames[2]["live_jitter"]["position_jitter_mm"] is None
    assert frames[0]["live_jitter"]["elapsed_s"] == pytest.approx(frames[0]["elapsed_ms"] / 1000.)
    assert worker.last_summary["status"] == "complete"
    assert worker.last_summary["frames_recorded"] == 8
    with (directory / "jitter.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 8
    assert rows[2]["position_jitter_mm"] == ""
    assert rows[2]["pose_available"] == "False"
    assert rows[2]["position_x_mm"] == "" and rows[2]["position_z_mm"] == ""
    assert float(rows[0]["position_z_mm"]) == pytest.approx(frames[0]["pose"]["position_mm"][2])
    with (directory / "markers.csv").open(newline="") as source:
        marker_rows = list(csv.DictReader(source))
    assert len(marker_rows) == sum(len(frame["marker_coordinates"]) for frame in frames)
    assert all(row["position_z_mm"] for row in marker_rows)
    assert json.loads((directory / "session.json").read_text())["coordinates"]["frame"] == "camera optical frame"


def test_stop_recording_keeps_preview_running(worker_setup, tmp_path):
    worker, camera, _ = worker_setup

    def action(index):
        if index == 4:
            worker.stop_recording()
        if index == 7:
            worker.disconnect()
    camera.callback = action
    worker.start_recording(tmp_path / "run")
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive() and camera.closed
    assert camera.index == 7
    assert worker.last_summary["frames_recorded"] == 4
    assert worker.last_summary["status"] == "complete"


def test_camera_error_finalizes_ui_recording(worker_setup, tmp_path):
    worker, camera, _ = worker_setup
    worker.args.reconnect = False

    def action(index):
        if index == 5:
            raise CameraError("unplugged")
    camera.callback = action
    worker.start_recording(tmp_path / "partial")
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive() and camera.closed
    assert worker.last_summary["status"] == "error"
    assert worker.last_summary["frames_recorded"] == 4
    assert ("error", "unplugged") in events(worker)


def test_interrupt_finalizes_partial_recording(worker_setup, tmp_path):
    worker, camera, _ = worker_setup
    camera.callback = lambda index: worker.disconnect(KeyboardInterrupt("SIGINT")) if index == 3 else None
    worker.start_recording(tmp_path / "interrupted")
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive() and camera.closed
    assert worker.last_summary["status"] == "interrupted"
    assert worker.last_summary["frames_recorded"] == 3


def test_disconnect_during_warmup_does_not_start_recording(worker_setup, tmp_path):
    worker, camera, _ = worker_setup
    worker.args.warmup_seconds = 10
    camera.callback = lambda index: worker.disconnect() if index == 1 else None
    worker.start_recording(tmp_path / "not-started")
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive() and camera.closed
    assert camera.index == 1
    assert not (tmp_path / "not-started").exists()


def test_recording_error_does_not_overwrite_or_disconnect_preview(worker_setup, tmp_path):
    worker, camera, _ = worker_setup
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("keep")
    camera.callback = lambda index: worker.disconnect() if index == 4 else None
    worker.start_recording(existing)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive() and camera.closed
    assert camera.index == 4
    assert marker.read_text() == "keep"
    assert any(kind == "record_error" for kind, _ in events(worker))
    assert worker.last_summary is None


def test_gui_recording_buttons_save_and_release_camera(worker_setup, tmp_path):
    try:
        import tkinter as tk
    except ImportError:
        pytest.skip("Tkinter is not installed")
    pytest.importorskip("PIL.ImageTk")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("No desktop display available")
    from charuco_realsense.gui import MonitorApp

    worker, camera, args = worker_setup
    root.withdraw()
    app = MonitorApp(root, args, ["front", "rear"], worker_factory=lambda args: worker, auto_connect=False)

    def until(condition):
        deadline = time.monotonic() + 5
        while not condition() and time.monotonic() < deadline:
            root.update()
            time.sleep(.01)
        assert condition()

    try:
        app.connect_button.invoke()
        until(lambda: app.connected)
        until(lambda: app.last_report is not None and app.last_report["pose"] is not None)
        assert len(app.coordinate_table.get_children()) == len(app.last_report["marker_coordinates"])
        assert float(app.metrics["z"].get()) == pytest.approx(app.last_report["pose"]["position_mm"][2], abs=.01)
        app.copy_button.invoke()
        assert "camera frame / mm" in root.clipboard_get()
        assert "Quality passed:" in root.clipboard_get()
        missing = copy.deepcopy(app.last_report)
        missing.update(pose=None, pose_available=False, pose_quality_passed=False, board_recognized=False,
                       marker_coordinates=[], matched_marker_ids=[], detected_charuco_corner_count=0)
        app._show_preview({"report": missing, "image": app.last_image, "points": [], "recording": False})
        assert app.metrics["z"].get() == "--" and app.copy_button.instate(["disabled"])
        assert not app.coordinate_table.get_children()
        app.record_button.invoke()
        until(lambda: app.recording and worker.recorded_frames >= 4)
        app.stop_button.invoke()
        until(lambda: not app.recording and app.last_session is not None)
        path = app.last_session
        assert json.loads((Path(path) / "summary.json").read_text())["frames_recorded"] >= 4
        assert "Saved" in app.record_var.get()
        app.close()
        until(lambda: not worker.is_alive())
        assert camera.closed
    finally:
        worker.disconnect()
        if worker.ident is not None:
            worker.join(timeout=5)
        try:
            root.destroy()
        except tk.TclError:
            pass


def test_camera_loss_reconnects_and_resumes_recording_in_a_new_part(worker_setup, tmp_path):
    worker, camera, _ = worker_setup
    worker.args.reconnect_max_seconds = .05
    worker.args.hardware_reset_after = 2
    resets = []
    worker.device_reset = lambda args: resets.append(args.serial) or True
    opened = []
    original = worker.camera_factory

    def factory(args):
        opened.append(len(opened))
        if len(opened) in (2, 3):
            raise CameraError("No RealSense device found")  # Unplugged while recording.
        return original(args)
    worker.camera_factory = factory

    def action(index):
        if index == 5:
            raise CameraError("USB disconnected")
        if index == 9:
            worker.disconnect()
    camera.callback = action
    base = tmp_path / "series"
    worker.start_recording(base)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive() and camera.closed
    values = events(worker)
    kinds = [kind for kind, _ in values]
    lost = [payload for kind, payload in values if kind == "camera_lost"]
    assert [p["attempt"] for p in lost] == [1, 2, 3] and lost[0]["recording"]
    assert resets == [None]  # After the second consecutive failed attempt.
    assert kinds.count("ready") == 2 and kinds[-1] == "disconnected"
    started = [payload for kind, payload in values if kind == "record_started"]
    assert [(p["part"], p["reason"]) for p in started] == [(1, "start"), (2, "reconnect")]
    stopped = [payload for kind, payload in values if kind == "record_stopped"]
    assert [(p["final"], p["summary"]["status"]) for p in stopped] == [(False, "error"), (True, "complete")]
    second = tmp_path / "series-part0002"
    assert json.loads((second / "session.json").read_text())["segment"] == {
        "series": "series", "part": 2, "reason": "reconnect", "segment_minutes": 60.}
    first_frames = (base / "frames.jsonl").read_text().splitlines()
    second_frames = [json.loads(line) for line in (second / "frames.jsonl").read_text().splitlines()]
    assert len(first_frames) == 4 and len(second_frames) == 4
    assert second_frames[0]["frame_index"] == 0
    # Elapsed time continues across parts of one series.
    assert second_frames[0]["elapsed_ms"] > json.loads(first_frames[-1])["elapsed_ms"]
    assert worker.recorded_frames == 8
    assert worker.health["camera_losses"] == 1 and worker.health["reconnects"] == 1


def test_recording_rotates_into_parts(worker_setup, tmp_path):
    worker, camera, _ = worker_setup
    worker.args.segment_minutes = .05 / 60  # 50 ms parts; camera frames are 10 ms apart.
    camera.callback = lambda index: worker.disconnect() if index == 30 else None
    worker.start_recording(tmp_path / "rotating")
    worker.start()
    worker.join(timeout=5)
    parts = sorted(tmp_path.glob("rotating*"))
    assert len(parts) >= 3, events(worker)
    total = 0
    for part in parts:
        summary = json.loads((part / "summary.json").read_text())
        assert summary["status"] == "complete"
        total += summary["frames_recorded"]
        with (part / "jitter.csv").open(newline="") as source:
            assert len(list(csv.DictReader(source))) == summary["frames_recorded"]
    assert total == worker.recorded_frames == 30
    reasons = [payload["reason"] for kind, payload in events(worker) if kind == "record_started"]
    assert reasons[0] == "start" and set(reasons[1:]) == {"rotation"}


def test_low_disk_space_stops_recording_but_keeps_preview(worker_setup, tmp_path, monkeypatch):
    from charuco_realsense import monitor
    worker, camera, _ = worker_setup
    free = [10e9]
    monkeypatch.setattr(monitor, "_free_bytes", lambda path: free[0])

    def action(index):
        if index == 3:
            free[0] = 1e9
            worker._next_disk_check = 0.
        if index == 6:
            worker.disconnect()
    camera.callback = action
    worker.start_recording(tmp_path / "disk")
    worker.start()
    worker.join(timeout=5)
    values = events(worker)
    assert any(kind == "record_error" and "Low disk space" in payload for kind, payload in values)
    assert worker.last_summary["status"] == "error" and worker.last_summary["frames_recorded"] == 3
    assert camera.index == 6 and worker.request is None


def test_storage_error_stops_recording_but_keeps_preview(worker_setup, tmp_path):
    worker, camera, _ = worker_setup

    def action(index):
        if index == 3:
            worker.recorder._file.close()  # Next write fails like a lost disk.
        if index == 6:
            worker.disconnect()
    camera.callback = action
    worker.start_recording(tmp_path / "storage")
    worker.start()
    worker.join(timeout=5)
    assert camera.index == 6
    assert any(kind == "record_error" for kind, _ in events(worker))
    assert worker.recorder is None and worker.request is None


def test_frame_processing_error_is_skipped(worker_setup, tmp_path, monkeypatch):
    from charuco_realsense import monitor
    worker, camera, _ = worker_setup
    original = monitor._analyze
    calls = []

    def flaky(*args):
        calls.append(1)
        if len(calls) == 2:
            raise cv2.error("corrupt frame")
        return original(*args)
    monkeypatch.setattr(monitor, "_analyze", flaky)
    camera.callback = lambda index: worker.disconnect() if index == 5 else None
    worker.start_recording(tmp_path / "skip")
    worker.start()
    worker.join(timeout=5)
    assert worker.last_summary["frames_recorded"] == 4
    assert worker.health["processing_errors"] == 1
    assert any(kind == "warning" for kind, _ in events(worker))


def test_stop_while_camera_unavailable_ends_series(worker_setup, tmp_path):
    worker, camera, _ = worker_setup
    worker.args.reconnect_max_seconds = 5.
    worker.args.hardware_reset_after = 0

    def factory(args):
        worker.stop_recording()
        raise CameraError("No RealSense device found")
    worker.camera_factory = factory
    worker.start_recording(tmp_path / "never")
    worker.start()
    values = []
    deadline = time.monotonic() + 2
    while not any(kind == "record_stopped" for kind, _ in values) and time.monotonic() < deadline:
        values.append(worker.events.get(timeout=2))
    worker.disconnect()
    worker.join(timeout=2)
    assert not worker.is_alive()
    values += events(worker)
    assert [kind for kind, _ in values] == ["camera_lost", "record_stopped", "disconnected"]
    assert ("record_stopped", {"path": None, "summary": None, "part": 0, "reason": "stopped", "final": True}) in values
    assert not (tmp_path / "never").exists()
