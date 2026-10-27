"""Linux RealSense client cleanup. Standard library only; no camera imports."""
import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import signal
import sys
import time


PROJECT = Path(__file__).resolve().parents[2]
SDK_TOOLS = {"realsense-viewer", "realsense-depth-quality", "rs-capture", "rs-convert",
             "rs-enumerate-devices", "rs-fw-update", "rs-record", "rs-save-to-disk",
             "rs-multicam", "rs-pointcloud", "rs-align", "rs-depth", "rs-color",
             "rs-hello-realsense", "rs-pose", "rs-pose-and-image"}


@dataclass(frozen=True)
class Process:
    pid: int
    started: str
    command: tuple[str, ...]
    reason: str


def _stat(pid, proc=Path("/proc")):
    # comm can contain spaces and parentheses; fields after its final ')' are fixed.
    fields = (proc / str(pid) / "stat").read_text().rpartition(")")[2].split()
    return fields[0], int(fields[1]), fields[19]  # state, PPID, start ticks


def _ancestors(proc):
    result = set()
    pid = os.getpid()
    while pid > 1 and pid not in result:
        result.add(pid)
        try:
            _, pid, _ = _stat(pid, proc)
        except (OSError, ValueError, IndexError):
            break
    return result


def _command_reason(command, cwd):
    if not command:
        return None
    name = Path(command[0]).name
    if name in SDK_TOOLS:
        return "RealSense SDK tool"
    if name == "charuco-realsense":
        return "ChArUco camera client"
    if re.fullmatch(r"(?:python|python3(?:\.\d+)?|pypy3)", name):
        arguments = iter(command[1:])
        for argument in arguments:
            if argument == "-c":
                break
            if argument == "-m":
                if next(arguments, "") in {"charuco_realsense", "charuco_realsense.cli"}:
                    return "ChArUco camera client"
                break
            if argument in {"-W", "-X"}:
                next(arguments, None)
                continue
            if argument.startswith("-"):
                continue
            if Path(argument).name == "main.py" and (cwd / argument).resolve() == PROJECT / "main.py":
                return "ChArUco camera client"
            break  # Only the executed script counts, never its arbitrary arguments.
    if name in {"bash", "sh", "start_camera.sh"}:
        script = command[0] if name == "start_camera.sh" else next((arg for arg in command[1:] if not arg.startswith("-")), "")
        if Path(script).name == "start_camera.sh" and (cwd / script).resolve() == PROJECT / "start_camera.sh":
            return "ChArUco launcher"
    return None


def discover(proc=Path("/proc")):
    """Only same-user clients; exclude this command and its caller ancestry.

    Inspect argv tokens and loaded SDK libraries, never arbitrary command-line
    substrings (which could accidentally match editors, terminals or this script).
    """
    excluded = _ancestors(proc)
    result = []
    for directory in proc.iterdir():
        if not directory.name.isdigit() or int(directory.name) in excluded:
            continue
        try:
            if directory.stat().st_uid != os.geteuid():
                continue
            pid = int(directory.name)
            state, _, started = _stat(pid, proc)
            if state == "Z":
                continue
            command = tuple(s.decode(errors="replace") for s in
                            (directory / "cmdline").read_bytes().split(b"\0") if s)
            try:
                cwd = (directory / "cwd").resolve()
                reason = _command_reason(command, cwd)
            except OSError:
                reason = None
            if reason is None:
                try:
                    maps = (directory / "maps").read_text()
                except OSError:
                    maps = ""
                if re.search(r"/(?:pyrealsense2[^/\s]*|librealsense2[^/\s]*)\.so(?:[.\s]|$)", maps):
                    reason = "Loaded RealSense SDK"
            if reason:
                result.append(Process(pid, started, command, reason))
        except (OSError, ValueError, IndexError):
            continue  # Process exited during the snapshot or cannot be inspected.
    return sorted(result, key=lambda p: p.pid)


def _alive(process):
    try:
        state, _, started = _stat(process.pid)
        return state != "Z" and started == process.started
    except (OSError, ValueError, IndexError):
        return False


def _send(process, number):
    if not _alive(process):
        return
    # A pidfd remains attached to the original process even if its PID is reused.
    if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
        descriptor = os.pidfd_open(process.pid)
        try:
            if _alive(process):
                signal.pidfd_send_signal(descriptor, number)
        finally:
            os.close(descriptor)
    else:
        os.kill(process.pid, number)


def stop(processes, grace=5., force=False):
    remaining = list(processes)
    phases = [(signal.SIGKILL, 1.)] if force else [
        (signal.SIGINT, grace), (signal.SIGTERM, 2.), (signal.SIGKILL, 1.)]
    for number, timeout in phases:
        remaining = [p for p in remaining if _alive(p)]
        if not remaining:
            break
        for process in remaining:
            try:
                _send(process, number)
                print(f"{signal.Signals(number).name} -> PID {process.pid}", flush=True)
            except ProcessLookupError:
                pass
            except OSError as exc:
                print(f"Cannot signal PID {process.pid}: {exc}", file=sys.stderr)
        deadline = time.monotonic() + timeout
        while remaining and time.monotonic() < deadline:
            time.sleep(.05)
            remaining = [p for p in remaining if _alive(p)]
    return [p for p in remaining if _alive(p)]


def main(argv=None):
    parser = argparse.ArgumentParser(prog="kill.sh", description="Stop all inspectable RealSense clients owned by the current user (Linux).")
    parser.add_argument("--dry-run", action="store_true", help="List matching clients without stopping them")
    parser.add_argument("--force", action="store_true", help="Immediately use SIGKILL; recordings cannot be finalized")
    parser.add_argument("--grace", type=float, default=5., help="Seconds to allow graceful saving before escalation (default: 5)")
    args = parser.parse_args(argv)
    if not 0 <= args.grace <= 300:
        parser.error("--grace must be between 0 and 300 seconds")
    if not Path("/proc/self/stat").exists():
        parser.error("This cleanup script requires Linux /proc")
    clients = discover()
    if not clients:
        print("No RealSense clients found for the current user.")
        return 0
    for process in clients:
        executable = process.command[0] if process.command else "unknown executable"
        print(f"PID {process.pid}: {executable} [{process.reason}]", flush=True)
    if args.dry_run:
        return 0
    remaining = stop(clients, args.grace, args.force)
    if remaining:
        print("Still running: " + ", ".join(str(p.pid) for p in remaining), file=sys.stderr)
        return 1
    print(f"Stopped {len(clients)} RealSense client(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
