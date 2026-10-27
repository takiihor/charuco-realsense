"""Headless 24/7 service, systemd notify protocol, and the GUI's reconnect states."""
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from charuco_realsense.camera import Frame
from charuco_realsense.cli import parser
from charuco_realsense.monitor import CaptureWorker
from charuco_realsense.service import Watchdog, configure_logging, run_service


@pytest.fixture
def notify_socket(tmp_path, monkeypatch):
    path = str(tmp_path / "notify")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(path)
    server.settimeout(.3)
    monkeypatch.setenv("NOTIFY_SOCKET", path)
    monkeypatch.setenv("WATCHDOG_USEC", "200000")
    yield server
    server.close()


def received(server):
    try:
        return server.recv(4096).decode()
    except socket.timeout:
        return None


def test_watchdog_pings_only_while_the_capture_loop_progresses(notify_socket):
    watchdog = Watchdog(stall_seconds=5.)
    assert watchdog.enabled and watchdog.interval == pytest.approx(.5)
    watchdog.ready("Connecting camera")
    assert received(notify_socket) == "READY=1\nSTATUS=Connecting camera"
    worker = SimpleNamespace(is_alive=lambda: True, heartbeat=time.monotonic())
    watchdog.tick(worker)
    assert received(notify_socket) == "WATCHDOG=1"
    watchdog.tick(worker)
    assert received(notify_socket) is None  # Rate-limited to half the systemd interval.
    worker.heartbeat -= 60.
    watchdog._last_ping = 0.
    assert watchdog.worker_stalled(worker) == pytest.approx(60., abs=1.)
    watchdog.tick(worker)
    assert received(notify_socket) is None  # A hung capture thread lets systemd restart us.
    watchdog.status("Camera live")
    watchdog.status("Camera live")
    assert received(notify_socket) == "STATUS=Camera live"
    assert received(notify_socket) is None


def test_watchdog_is_a_noop_when_unsupervised(monkeypatch):
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    watchdog = Watchdog()
    assert not watchdog.enabled
    watchdog.ready()
    watchdog.tick(None)


class Camera:
    intrinsics = {"fx": 600., "fy": 600., "ppx": 320., "ppy": 240., "width": 640, "height": 480,
                  "model": "none", "coeffs": [0.] * 5}
    metadata = {"intrinsics": intrinsics, "stream_info": "Service test camera"}

    def __init__(self):
        self.index = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def read(self):
        time.sleep(.01)
        self.index += 1
        return Frame(np.full((480, 640, 3), 255, np.uint8), self.intrinsics, self.index * 10., self.index)


def test_service_records_until_sigterm_and_logs(artifacts, tmp_path):
    directory, _ = artifacts
    args = parser().parse_args(["service", "--board", "front", "--manifest", str(directory / "board_manifest.json"),
                                "--warmup-seconds", "0", "--output-dir", str(tmp_path)])
    camera = Camera()
    workers = []

    def factory(args):
        workers.append(CaptureWorker(args, camera_factory=lambda args: camera))
        return workers[0]

    def terminate():
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not (workers and workers[0].recorded_frames >= 3):
            time.sleep(.01)
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=terminate, daemon=True).start()
    try:
        assert run_service(args, worker_factory=factory) == 128 + signal.SIGTERM
    finally:
        configure_logging(None, stderr=False)  # Detach the test's log file handler.
    assert camera.closed and not workers[0].is_alive()
    summary = json.loads(next(tmp_path.glob("run-*/summary.json")).read_text())
    assert summary["status"] == "interrupted" and summary["frames_recorded"] >= 3
    text = (tmp_path / "logs" / "charuco-realsense.log").read_text()
    for expected in ("Camera ready", "Recording part 1 (start)", "SIGTERM received", "Saved part 1", "Camera released"):
        assert expected in text


@pytest.fixture
def app(tmp_path):
    tk = pytest.importorskip("tkinter")
    pytest.importorskip("PIL.ImageTk")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("No desktop display available")
    root.withdraw()
    from charuco_realsense.gui import MonitorApp
    args = parser().parse_args(["gui", "--board", "front", "--output-dir", str(tmp_path)])
    monitor = MonitorApp(root, args, ["front"], auto_connect=False)
    yield monitor
    if not monitor.destroyed:
        root.destroy()


def test_gui_follows_camera_loss_and_resumed_recording(app):
    ready = {"camera": {"stream_info": "Test"}, "board": {}, "reconnected": False}
    app._event("ready", ready)
    app._event("record_started", {"path": "/data/run-1", "part": 1, "reason": "start"})
    assert app.recording and app.status_var.get() == "Camera live"
    assert app.record_button.instate(["disabled"]) and not app.stop_button.instate(["disabled"])
    app._event("record_stopped", {"path": "/data/run-1", "summary": None, "part": 1, "reason": "reconnect", "final": False})
    app._event("camera_lost", {"error": "USB disconnected", "attempt": 1, "retry_in": 2., "recording": True})
    assert app.recording and not app.connected and app.reconnecting
    assert app.status_var.get() == "Reconnecting" and app.rec_pill_var.get() == "REC PAUSED"
    assert "Paused" in app.record_var.get() and "new part" in app.saved_var.get()
    assert app.record_button.instate(["disabled"]) and not app.stop_button.instate(["disabled"])
    app._event("ready", {**ready, "reconnected": True})
    app._event("record_started", {"path": "/data/run-1-part0002", "part": 2, "reason": "reconnect"})
    assert app.connected and not app.reconnecting and app.record_part == 2
    assert "reconnected" in app.saved_var.get()
    summary = {"frames_recorded": 12, "status": "complete", "error": None, "rates": {"board_recognized": .5}}
    app._event("record_stopped", {"path": "/data/run-1-part0002", "summary": summary, "part": 2, "reason": "stopped", "final": True})
    assert not app.recording and app.record_var.get() == "Saved 12 frames in part 2"
    assert app.last_session == "/data/run-1-part0002"
    assert not app.record_button.instate(["disabled"]) and not app.rec_pill.winfo_manager()
    messages = [app.event_table.item(item, "values")[2] for item in app.event_table.get_children()]
    assert any("Camera unavailable (attempt 1)" in message for message in messages)
    assert messages[0].startswith("Saved part 2")


def test_gui_poll_survives_ui_errors(app, monkeypatch):
    def broken(now):
        raise RuntimeError("drawing failed")
    monkeypatch.setattr(app, "_update_health", broken)
    for callback in app.root.tk.call("after", "info"):
        app.root.after_cancel(callback)
    app._poll()
    assert app.root.tk.call("after", "info"), "the poll loop must reschedule itself after an error"


def test_warmup_keeps_the_heartbeat_fresh(artifacts, tmp_path):
    directory, _ = artifacts
    args = parser().parse_args(["service", "--board", "front", "--manifest", str(directory / "board_manifest.json"),
                                "--warmup-seconds", "0.3", "--output-dir", str(tmp_path)])
    camera = Camera()
    worker = CaptureWorker(args, camera_factory=lambda args: camera)
    beats = []
    original = camera.read

    def read():
        beats.append(worker.heartbeat)
        return original()
    camera.read = read
    worker.start()
    deadline = time.monotonic() + 5
    while worker.health["frames"] == 0 and time.monotonic() < deadline:
        time.sleep(.01)
    worker.disconnect()
    worker.join(timeout=5)
    warmup = beats[:10]
    assert len(set(warmup)) == len(warmup), "every warmup read must refresh the heartbeat"


def test_service_exits_when_the_camera_never_releases(artifacts, tmp_path):
    """A pipeline stop stuck in the USB stack must not keep the service alive forever."""
    directory, _ = artifacts
    ready = tmp_path / "ready"
    script = r'''
import pathlib, sys, threading, time
import numpy as np
from charuco_realsense import service
from charuco_realsense.camera import Frame
from charuco_realsense.cli import main
from charuco_realsense.monitor import CaptureWorker

manifest, output, ready = map(pathlib.Path, sys.argv[1:])
service.CLOSE_GRACE_SECONDS = .5
class Camera:
    intrinsics = {"fx": 600., "fy": 600., "ppx": 320., "ppy": 240., "width": 640, "height": 480, "model": "none", "coeffs": [0.] * 5}
    metadata = {"intrinsics": intrinsics, "stream_info": "Hung camera"}
    index = 0
    def __enter__(self): return self
    def __exit__(self, *args): threading.Event().wait()  # Never returns, like a hung USB close.
    def read(self):
        time.sleep(.01)
        self.index += 1
        if self.index == 5: ready.touch()
        return Frame(np.full((480, 640, 3), 255, np.uint8), self.intrinsics, self.index * 10., self.index)
original = service.run_service
service.run_service = lambda args: original(args, worker_factory=lambda args: CaptureWorker(args, camera_factory=lambda args: Camera()))
raise SystemExit(main(["service", "--board", "front", "--manifest", str(manifest), "--warmup-seconds", "0", "--output-dir", str(output)]))
'''
    child = subprocess.Popen([sys.executable, "-c", script, str(directory / "board_manifest.json"), str(tmp_path), str(ready)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        deadline = time.monotonic() + 8
        while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(.02)
        assert ready.exists(), child.communicate(timeout=2)
        os.kill(child.pid, signal.SIGTERM)
        _, stderr = child.communicate(timeout=10)
        assert child.returncode == 128 + signal.SIGTERM, stderr
        assert "exiting without waiting" in stderr
        summary = json.loads(next(tmp_path.glob("run-*/summary.json")).read_text())
        assert summary["status"] == "interrupted"
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=2)
