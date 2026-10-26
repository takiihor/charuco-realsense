import csv
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
