"""Real OS signals must close the Tk loop and its non-daemon camera worker."""
import json
import os
import signal
import subprocess
import sys
import time

import pytest


@pytest.mark.parametrize("number", [signal.SIGINT, signal.SIGTERM])
def test_gui_signal_saves_recording_and_exits(artifacts, tmp_path, number):
    tk = pytest.importorskip("tkinter")
    pytest.importorskip("PIL.ImageTk")
    try:
        root = tk.Tk()
        root.destroy()
    except tk.TclError:
        pytest.skip("No desktop display available")
    directory, _ = artifacts
    ready = tmp_path / "ready"
    closed = tmp_path / "closed"
    # Exercise the actual CLI, signal context, Tk mainloop, worker and recorder.
    # Only the USB camera is replaced; the child runs in its own process group.
    script = r'''
import pathlib, sys, time
import numpy as np
from charuco_realsense.camera import Frame
from charuco_realsense.cli import main
from charuco_realsense import gui
from charuco_realsense.monitor import CaptureWorker

manifest, output, ready, closed = map(pathlib.Path, sys.argv[1:])
class Camera:
    intrinsics = {"fx": 600., "fy": 600., "ppx": 320., "ppy": 240., "width": 640, "height": 480, "model": "none", "coeffs": [0.] * 5}
    metadata = {"intrinsics": intrinsics, "stream_info": "Signal test camera"}
    def __init__(self): self.index = 0
    def __enter__(self): return self
    def __exit__(self, *args): closed.touch()
    def read(self):
        time.sleep(.02)
        self.index += 1
        return Frame(np.full((480, 640, 3), 255, np.uint8), self.intrinsics, self.index * 20., self.index)
actual_app = gui.MonitorApp
def create_app(root, args, names, **kwargs):
    app = actual_app(root, args, names, worker_factory=lambda args: CaptureWorker(args, camera_factory=lambda args: Camera()), **kwargs)
    def record():
        if app.connected and not app.recording and not app.pending_record: app.start_recording()
        if app.worker and app.worker.recorded_frames >= 3:
            ready.touch()
        else: root.after(20, record)
    root.after(50, record)
    return app
gui.MonitorApp = create_app
raise SystemExit(main(["gui", "--board", "front", "--manifest", str(manifest), "--warmup-seconds", "0", "--output-dir", str(output)]))
'''
    child = subprocess.Popen([sys.executable, "-c", script, str(directory / "board_manifest.json"),
                              str(tmp_path), str(ready), str(closed)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        deadline = time.monotonic() + 8
        while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(.02)
        assert ready.exists(), child.communicate(timeout=2) if child.poll() is not None else "GUI did not start recording"
        os.kill(child.pid, number)
        stdout, stderr = child.communicate(timeout=4)
        assert child.returncode == 128 + number, (stdout, stderr)
        assert closed.exists(), "Camera must be released before exit"
        summary = json.loads(next(tmp_path.glob("run-*/summary.json")).read_text())
        assert summary["status"] == "interrupted"
        assert summary["frames_recorded"] >= 3
        assert "KeyboardInterrupt" not in stderr and "Traceback" not in stderr
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=2)
