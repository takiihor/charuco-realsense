"""Unattended operation: rotating logs, systemd watchdog, and a headless recorder."""
from datetime import datetime
import logging
import logging.handlers
import os
from pathlib import Path
from queue import Empty
import signal
import socket
import sys
import time

from .shutdown import SignalStop

log = logging.getLogger("charuco_realsense")


def configure_logging(log_file: str | None, stderr=True) -> Path | None:
    """INFO to stderr (journald when supervised) and a size-bounded rotating file."""
    log.setLevel(logging.INFO)
    for handler in list(log.handlers):
        log.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(threadName)s: %(message)s")
    if stderr:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        log.addHandler(handler)
    path = None
    if log_file:
        path = Path(log_file).expanduser()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.handlers.RotatingFileHandler(path, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
        except OSError as exc:
            log.warning("Cannot write log file %s: %s", path, exc)
            path = None
        else:
            handler.setFormatter(formatter)
            log.addHandler(handler)
    log.propagate = False
    return path


def default_log_file(args) -> str | None:
    if args.log_file is not None:
        return args.log_file or None  # An empty string disables the file.
    return str(Path(args.output_dir).expanduser() / "logs" / "charuco-realsense.log")


def recording_path(output_dir) -> Path:
    return Path(output_dir).expanduser() / ("run-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))


class Watchdog:
    """systemd sd_notify protocol (READY/STATUS/WATCHDOG); a no-op when unsupervised.

    Pings only while the camera worker reports progress, so a capture thread stuck
    inside the SDK or USB stack lets systemd restart the whole process.
    """

    def __init__(self, stall_seconds=30.):
        self.address = os.environ.get("NOTIFY_SOCKET")
        if self.address and self.address.startswith("@"):
            self.address = "\0" + self.address[1:]
        usec = os.environ.get("WATCHDOG_USEC")
        self.interval = max(.5, int(usec) / 2e6) if usec and usec.isdigit() else 5.
        self.stall_seconds = stall_seconds
        self._last_ping = 0.
        self._status = None

    @property
    def enabled(self):
        return bool(self.address)

    def _send(self, message):
        if not self.address:
            return
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM | socket.SOCK_CLOEXEC) as sock:
                sock.sendto(message.encode(), self.address)
        except OSError as exc:
            log.debug("sd_notify failed: %s", exc)

    def ready(self, status="Running"):
        self._send(f"READY=1\nSTATUS={status}")
        self._status = status

    def status(self, text):
        if text != self._status:
            self._status = text
            self._send(f"STATUS={text}")

    def stopping(self):
        self._send("STOPPING=1")

    def worker_stalled(self, worker, now=None):
        """Seconds without capture-loop progress, or None while healthy/idle."""
        if worker is None or not worker.is_alive():
            return None
        age = (now if now is not None else time.monotonic()) - worker.heartbeat
        return age if age > self.stall_seconds else None

    def tick(self, worker):
        now = time.monotonic()
        if now - self._last_ping < self.interval or self.worker_stalled(worker, now) is not None:
            return
        self._last_ping = now
        self._send("WATCHDOG=1")


def describe_event(kind, payload):
    """(logging level, one-line message) for a worker event; None for no entry."""
    if kind == "ready":
        prefix = "Camera reconnected" if payload.get("reconnected") else "Camera ready"
        return logging.INFO, f"{prefix}: {payload['camera'].get('stream_info', 'color stream')}"
    if kind == "record_started":
        return logging.INFO, f"Recording part {payload['part']} ({payload['reason']}) to {payload['path']}"
    if kind == "record_stopped":
        summary = payload["summary"]
        if payload["path"] is None:
            return logging.INFO, "Recording stopped while the camera was unavailable"
        if summary is None:
            return logging.WARNING, f"Part {payload['part']} saved without a summary: {payload['path']}"
        rate = summary["rates"]["board_recognized"]
        detected = f"{rate:.1%}" if rate is not None else "n/a"
        return (logging.INFO if summary["status"] == "complete" else logging.WARNING,
                f"Saved part {payload['part']} ({payload['reason']}, {summary['status']}): "
                f"{summary['frames_recorded']} frames, board detected {detected} -> {payload['path']}")
    if kind == "camera_lost":
        return logging.WARNING, (f"Camera unavailable (attempt {payload['attempt']}): {payload['error']}; "
                                 f"retrying in {payload['retry_in']:.0f} s")
    if kind == "record_error":
        return logging.ERROR, f"Recording stopped: {payload}"
    if kind == "warning":
        return logging.WARNING, str(payload)
    if kind == "error":
        return logging.ERROR, f"Camera stopped: {payload}"
    if kind == "disconnected":
        return logging.INFO, "Camera released"
    return None


def log_event(kind, payload):
    described = describe_event(kind, payload)
    if described is not None:
        log.log(*described)
    return described


STATUS_TEXT = {"ready": "Camera live", "record_started": "Recording", "camera_lost": "Reconnecting camera",
               "error": "Camera error", "record_error": "Recording stopped", "disconnected": "Camera released"}


def run_service(args, worker_factory=None) -> int:
    """Record without a display until SIGINT/SIGTERM; reconnect and rotate parts."""
    from .monitor import CaptureWorker
    log_path = configure_logging(default_log_file(args))
    if log_path:
        log.info("Logging to %s", log_path)
    worker = (worker_factory or CaptureWorker)(args)
    if not args.no_record:
        worker.start_recording(recording_path(args.output_dir), args.save_frames)
    watchdog = Watchdog(args.stall_seconds)
    fatal = None
    with SignalStop() as shutdown:
        worker.start()
        watchdog.ready("Connecting camera")
        log.info("Service started for board %r; Ctrl+C or SIGTERM saves and exits", args.board)
        stop_sent = False
        stop_at = None
        abandoned = False
        next_stall_log = 0.
        while worker.is_alive() or not worker.events.empty():
            if shutdown.requested.is_set() and not stop_sent:
                stop_sent = True
                stop_at = time.monotonic()
                name = signal.Signals(shutdown.signal_number).name
                log.info("%s received; saving and releasing the camera", name)
                watchdog.stopping()
                worker.disconnect(KeyboardInterrupt(f"Stopped by {name}"))
            try:
                kind, payload = worker.events.get(timeout=.25)
            except Empty:
                kind = None
            if kind is not None:
                log_event(kind, payload)
                if kind == "error":
                    fatal = payload
                if kind in STATUS_TEXT:
                    watchdog.status(STATUS_TEXT[kind])
            stalled = watchdog.worker_stalled(worker)
            if stalled is not None and time.monotonic() >= next_stall_log:
                next_stall_log = time.monotonic() + 30.
                log.error("Capture thread has made no progress for %.0f s; the USB device may be hung", stalled)
            if not stop_sent:
                watchdog.tick(worker)
            elif worker.is_alive() and _give_up_waiting(worker, time.monotonic() - stop_at):
                log.error("Camera did not release within %.0f s; exiting without waiting for it",
                          time.monotonic() - stop_at)
                abandoned = True
                break
        if not abandoned:
            worker.join()
    code = 128 + shutdown.signal_number if shutdown.signal_number is not None else 2 if fatal else 0
    if abandoned:
        # The capture thread is stuck in the USB stack after its recording was
        # saved; a normal interpreter exit would wait for it forever.
        logging.shutdown()
        os._exit(code)
    return code


CLOSE_GRACE_SECONDS = 20.
CLOSE_LIMIT_SECONDS = 120.


def _give_up_waiting(worker, elapsed):
    """Recordings are finalized before the camera is released; then stop waiting for it."""
    return (worker.recorder is None and elapsed > CLOSE_GRACE_SECONDS) or elapsed > CLOSE_LIMIT_SECONDS
