import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from charuco_realsense import processes
from charuco_realsense.shutdown import SignalStop


def test_identify_clients_without_matching_editors_or_command_text():
    root = processes.PROJECT
    classify = lambda args: processes._command_reason(tuple(args), root)
    assert classify(["python3", "-u", "main.py", "gui", "--board", "front"])
    assert classify(["python", "-m", "charuco_realsense.cli", "stream"])
    assert classify(["rs-capture"])
    assert classify(["bash", str(root / "start_camera.sh")])
    assert classify(["code", str(root / "main.py")]) is None
    assert classify(["python3", "editor.py", str(root / "main.py")]) is None
    assert classify(["python3", "-c", "print('realsense')", str(root / "main.py")]) is None
    assert classify(["bash", "-c", "echo", str(root / "start_camera.sh")]) is None
    assert classify(["python", "main.py"]) is not None
    assert processes._command_reason(("python", "main.py"), Path("/tmp")) is None


def test_discover_loaded_sdk_libraries_and_exclude_zombies(tmp_path):
    def entry(pid, command, library="", state="S"):
        directory = tmp_path / str(pid)
        directory.mkdir()
        # Start ticks are field 22, after state + PPID + 17 intervening fields.
        (directory / "stat").write_text(f"{pid} (test process) {state} 1 " + "0 " * 17 + "12345\n")
        (directory / "cmdline").write_bytes(b"\0".join(s.encode() for s in command) + b"\0")
        (directory / "maps").write_text(library)
        (directory / "cwd").symlink_to(processes.PROJECT)
    entry(101, ["python3", "custom_tracker.py"], "/venv/lib/pyrealsense2/pyrealsense2.cpython-312-x86_64-linux-gnu.so\n")
    entry(102, ["custom-client"], "/usr/lib/librealsense2.so.2.55.1\n")
    entry(103, ["code", str(processes.PROJECT)])
    entry(104, ["realsense-viewer"], state="Z")
    assert [process.pid for process in processes.discover(tmp_path)] == [101, 102]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process cleanup")
def test_stop_escalates_only_the_identified_process(tmp_path):
    # This child deliberately ignores INT/TERM; the test signals only this PID.
    ready = tmp_path / "ready"
    code = "import signal,time,pathlib,sys; signal.signal(signal.SIGINT, signal.SIG_IGN); signal.signal(signal.SIGTERM, signal.SIG_IGN); pathlib.Path(sys.argv[1]).touch(); time.sleep(30)"
    child = subprocess.Popen([sys.executable, "-c", code, str(ready)])
    try:
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists()
        _, _, started = processes._stat(child.pid)
        target = processes.Process(child.pid, started, ("test-child",), "Test")
        assert not processes.stop([target], grace=.01)
        assert child.wait(timeout=2) == -signal.SIGKILL
        # A reused PID with a different start identity must never be signalled.
        stale = processes.Process(os.getpid(), "-1", ("stale",), "Test")
        assert not processes._alive(stale)
        processes._send(stale, signal.SIGKILL)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=2)


def test_signal_handler_records_shutdown_without_raising_and_restores_handlers():
    previous = signal.getsignal(signal.SIGINT)
    with SignalStop() as shutdown:
        os.kill(os.getpid(), signal.SIGINT)
        os.kill(os.getpid(), signal.SIGTERM)
        assert shutdown.requested.is_set()
        assert shutdown.signal_number == signal.SIGINT
    assert signal.getsignal(signal.SIGINT) == previous
