"""Detect boards, display a live stream, and record repeatability experiments."""
import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from . import __version__
from .calibration import load_calibration, opencv_calibration
from .camera import CameraError, RealsenseCamera, StreamConfig, list_devices
from .detect import CharucoDetector, Detection, report_for_board
from .metrics import load_reference_pose, pose_error
from .pose import PoseInputError, PoseOptions, estimate_pose
from .recording import SessionRecorder, summarize_recording, write_json


def _default_manifest(generator_path: str | None) -> str:
    output = Path(generator_path or "../charuco-generator") / "output"
    candidates = (
        output / "board_manifest.json",
        output / "two_stage" / "stage2_charuco_realsense_80mm" / "board_manifest.json",
    )
    for path in candidates:
        if path.is_file():
            return str(path)
    searched = ", ".join(str(path) for path in candidates)
    raise ValueError(f"No default ChArUco manifest found (checked {searched}). "
                     "Generate boards with charuco-generator, or pass --manifest PATH --board NAME.")


def _load_board(manifest_path: str, board_name: str | None, generator_path: str | None):
    if generator_path is not None:
        sys.path.insert(0, str(Path(generator_path) / "src"))
    try:
        from charuco_generator.board import reconstruct_board
        from charuco_generator.validation import load_manifest
    except ImportError as exc:
        raise ValueError("Install charuco-generator with --no-deps or supply --generator-path") from exc
    manifest = load_manifest(manifest_path)
    if not manifest.get("boards"):
        raise ValueError(f"No ChArUco boards in {manifest_path}; pass a ChArUco board manifest with --manifest")
    if board_name is None:
        names = [record["name"] for record in manifest["boards"]]
        board_name = "front" if "front" in names else names[0]
    record = next((b for b in manifest["boards"] if b["name"] == board_name), None)
    if record is None:
        names = ", ".join(b["name"] for b in manifest["boards"])
        raise ValueError(f"Unknown board {board_name!r} in {manifest_path}. Available boards: {names}")
    board = reconstruct_board(manifest, board_name)
    return board, record, manifest


def _pose_options(args):
    return PoseOptions(solver=args.solver, ransac=args.ransac, refine=args.refine,
                       min_corners=args.min_corners, min_coverage=args.min_coverage,
                       max_rmse_px=args.max_rmse_px, max_error_px=args.max_error_px,
                       min_inlier_fraction=args.min_inlier_fraction,
                       ransac_threshold_px=args.ransac_threshold_px)


def _analyze(board, record, detector, image, intrinsics, include_pose, options):
    started = time.perf_counter()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    detection = detector.detect(gray, intrinsics)
    pose = None
    pose_failure = "pose_unsolved" if include_pose else "pose_not_requested"
    if include_pose and intrinsics is not None:
        try:
            pose = estimate_pose(board, detection, intrinsics, options)
        except PoseInputError:
            pose_failure = "invalid_corner_correspondences"
    marker_counts = Counter(detection.marker_ids)
    if pose and any(marker_counts[i] > 1 for i in record["marker_ids"]):
        pose["quality_passed"] = False
        pose["quality_reasons"].append("duplicate_expected_marker_ids")
    recognized = bool(set(detection.marker_ids) & set(record["marker_ids"]))
    report = {
        "board": record["name"], "identifier": record["identifier"],
        "image_pixels": list(detection.image_pixels),
        **report_for_board(detection, record["marker_ids"], record["expected_charuco_corner_count"]),
        "board_recognized": recognized,
        "pose_requested": include_pose,
        "pose_available": pose is not None,
        "pose_quality_passed": pose["quality_passed"] if pose else False,
        "pose_quality_reasons": (pose["quality_reasons"] if pose else
                                 [pose_failure]),
        "pose": pose,
        "processing_ms": (time.perf_counter() - started) * 1000.,
        "passed": recognized,  # Preserve the existing recognition-only exit contract.
    }
    return detection, report


def _detect_image(board, record, image_bgr, border_bits, include_pose, intrinsics):
    """Compatibility wrapper for single-image callers."""
    return _analyze(board, record, CharucoDetector(board, border_bits), image_bgr,
                    intrinsics, include_pose, PoseOptions())[1]


def _draw(overlay: np.ndarray, detection: Detection, pose=None, intrinsics=None, labels=True) -> np.ndarray:
    """Marker outlines, corners and pose axes; labels=False leaves ID text to the caller."""
    for corner, marker_id in zip(detection.marker_corners, detection.marker_ids):
        pts = np.asarray(corner).reshape(4, 2).astype(int)
        cv2.polylines(overlay, [pts], True, (0, 0, 255), 2)
        if labels:
            cv2.putText(overlay, str(marker_id), (int(pts[0][0]), int(pts[0][1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    for corner, charuco_id in zip(detection.charuco_corners, detection.charuco_ids):
        x, y = (int(round(float(v))) for v in corner.ravel())
        cv2.circle(overlay, (x, y), 3, (0, 255, 0), -1)
        if labels:
            cv2.putText(overlay, str(charuco_id), (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    if pose is not None and intrinsics is not None and pose["positive_depth"]:
        matrix, coeffs = opencv_calibration(intrinsics, detection.image_pixels)
        cv2.drawFrameAxes(overlay, matrix, coeffs, np.asarray(pose["rvec"]),
                          np.asarray(pose["tvec_m"]), .03, 2)
    return overlay


def _open_camera(args, stop_event=None):
    return RealsenseCamera(serial=args.serial,
                           config=StreamConfig(args.width, args.height, args.fps, args.depth),
                           stop_event=stop_event)


def _settle_camera(camera, args, stop_event=None, progress=None):
    """Discard warmup frames; progress() runs after each read (worker heartbeat)."""
    deadline = time.monotonic() + args.warmup_seconds
    while time.monotonic() < deadline:
        if stop_event is not None and stop_event.is_set():
            return
        camera.read()
        if progress is not None:
            progress()
    if args.lock_exposure:
        if stop_event is not None and stop_event.is_set():
            return
        camera.lock_exposure()
        # Discard a few queued frames captured before the option took effect.
        for _ in range(3):
            if stop_event is not None and stop_event.is_set():
                return
            camera.read()


def _frame_fields(frame, index, host_time, started):
    return {"frame_index": index, "host_monotonic_ms": host_time * 1000.,
            "elapsed_ms": (host_time - started) * 1000.,
            "camera_timestamp_ms": frame.timestamp_ms,
            "camera_frame_number": frame.frame_number,
            "camera_timestamp_domain": frame.timestamp_domain}


def _cmd_detect(args) -> int:
    board, record, _ = _load_board(args.manifest, args.board, args.generator_path)
    detector = CharucoDetector(board, record["geometry"]["border_bits"])
    options = _pose_options(args)
    calibration = load_calibration(args.calibration) if args.calibration else None
    if args.image:
        if args.pose and calibration is None:
            raise ValueError("detect --image --pose requires --calibration")
        image = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image {args.image}")
        _, report = _analyze(board, record, detector, image, calibration, args.pose, options)
        report["source"] = args.image
        report["intrinsics"] = calibration
    else:
        with _open_camera(args) as camera:
            _settle_camera(camera, args)
            best = None
            started = time.monotonic()
            for index in range(args.frames):
                frame = camera.read()
                now = time.monotonic()
                intrinsics = calibration if calibration is not None else frame.intrinsics
                _, candidate = _analyze(board, record, detector, frame.bgr, intrinsics, args.pose, options)
                candidate.update(_frame_fields(frame, index, now, started))
                if best is None or candidate["detected_charuco_corner_count"] > best["detected_charuco_corner_count"]:
                    best = candidate
                    image = frame.bgr.copy()
            report = best
            report.update(source="realsense", frames_sampled=args.frames,
                          intrinsics=calibration if calibration is not None else camera.intrinsics,
                          camera=camera.metadata)
    if args.save_image and not cv2.imwrite(args.save_image, image):
        raise OSError(f"Cannot save analyzed image to {args.save_image}")
    if args.report:
        write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 1


def _metadata(args, camera, record, manifest, detector, options, intrinsics, reference):
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": {"charuco_realsense": __version__, "opencv": cv2.__version__, "numpy": np.__version__},
        "board": record,
        "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "camera": camera.metadata,
        "requested_stream": asdict(StreamConfig(args.width, args.height, args.fps, args.depth)),
        "effective_intrinsics": intrinsics,
        "calibration_source": args.calibration or "camera_factory",
        "detector": detector.settings,
        "pose_options": asdict(options),
        "warmup_seconds": args.warmup_seconds,
        "exposure_locked": args.lock_exposure,
        "reference_pose": reference,
        "recording": {"raw_poses": True, "smoothing": False, "save_frames": args.save_frames,
                      "seconds_limit": args.seconds, "frames_limit": args.frames,
                      "processing_ms_excludes": "capture, image saving, logging and GUI"},
    }


def _cmd_capture(args) -> int:
    board, record, manifest = _load_board(args.manifest, args.board, args.generator_path)
    options = _pose_options(args)
    calibration = load_calibration(args.calibration) if args.calibration else None
    reference = load_reference_pose(args.reference_pose) if args.reference_pose else None
    output = args.output if args.command == "benchmark" else args.record
    if args.save_frames and not output:
        raise ValueError("--save-frames requires --record (or benchmark --output)")
    if reference is not None and not output:
        raise ValueError("--reference-pose requires a recording")
    if output and Path(output).exists():
        raise ValueError(f"Recording directory already exists: {output}; choose a new directory")
    include_pose = args.pose or bool(output)
    show = args.command == "stream" or args.show
    detector = CharucoDetector(board, record["geometry"]["border_bits"])
    recorder = None
    try:
        with _open_camera(args) as camera:
            _settle_camera(camera, args)
            intrinsics = calibration if calibration is not None else camera.intrinsics
            opencv_calibration(intrinsics, (camera.intrinsics["width"], camera.intrinsics["height"]))
            if output:
                recorder = SessionRecorder(output, _metadata(args, camera, record, manifest,
                                                              detector, options, intrinsics, reference),
                                           save_frames=args.save_frames)
                print(f"Recording to {recorder.path}; keep camera and board stationary for jitter measurements", file=sys.stderr)
            if show:
                print(f"Streaming {record['name']}; press q to quit", file=sys.stderr)
            with (recorder if recorder is not None else nullcontext()):
                started = time.monotonic()
                last = None
                index = 0
                while (args.frames is None or index < args.frames) and (args.seconds is None or time.monotonic() - started < args.seconds):
                    frame = camera.read()
                    now = time.monotonic()
                    detection, report = _analyze(board, record, detector, frame.bgr, intrinsics,
                                                 include_pose, options)
                    report.update(_frame_fields(frame, index, now, started))
                    if reference is not None and report["pose"] is not None:
                        report["reference_error"] = pose_error(report["pose"], reference)
                    if recorder is not None:
                        recorder.add(report, frame.bgr)
                    if show:
                        fps = 1. / (now - last) if last is not None and now > last else 0.
                        overlay = _draw(frame.bgr.copy(), detection)
                        status = f"{record['name']}: {len(detection.marker_ids)} markers, {len(detection.charuco_ids)} corners, {fps:.0f} fps"
                        if report["pose"]:
                            pose = report["pose"]
                            status += f" | z={pose['position_mm'][2]:.1f} mm, error={pose['reprojection_rmse_px']:.2f} px"
                        color = (0, 255, 0) if (report["pose_quality_passed"] if include_pose else report["board_recognized"]) else (0, 0, 255)
                        cv2.putText(overlay, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
                        cv2.imshow("charuco-realsense", overlay)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    last = now
                    index += 1
    finally:
        if show:
            cv2.destroyAllWindows()
        if recorder is not None and recorder.summary is not None:
            print(f"Summary saved to {recorder.path / 'summary.json'}", file=sys.stderr)
    if recorder is not None:
        print(json.dumps(recorder.summary, indent=2, sort_keys=True, allow_nan=False))
        return 0 if recorder.summary["counts"]["board_recognized"] else 1
    return 0


def _positive_int(value):
    try:
        number = int(value)
        if number <= 0:
            raise ValueError
        return number
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc


def _finite_float(value):
    try:
        number = float(value)
        if not np.isfinite(number):
            raise ValueError
        return number
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be finite") from exc


def _nonnegative_int(value):
    try:
        number = int(value)
        if number < 0:
            raise ValueError
        return number
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from exc


def _positive_float(value):
    number = _finite_float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _nonnegative_float(value):
    number = _finite_float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def _fraction(value):
    number = _finite_float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="ChArUco recognition, pose quality and repeatability with RealSense")
    root.add_argument("--version", action="version", version=f"charuco-realsense {__version__}")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List connected RealSense devices")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", help="Board manifest (default: generator output, then two-stage RealSense 80 mm set)")
    common.add_argument("--generator-path", default=None)
    common.add_argument("--serial", default=None)
    common.add_argument("--width", type=_positive_int, default=1280)
    common.add_argument("--height", type=_positive_int, default=720)
    common.add_argument("--fps", type=_positive_int, default=30)
    common.add_argument("--depth", action="store_true", help="Also enable depth (unused by pose estimation)")
    common.add_argument("--warmup-seconds", type=_nonnegative_float, default=2.)
    common.add_argument("--lock-exposure", action="store_true", help="Freeze settled exposure; restore auto mode on exit")
    common.add_argument("--calibration", help="OpenCV-compatible intrinsics JSON; must match actual image size")
    common.add_argument("--pose", action="store_true", help="Estimate pose (automatic when recording)")
    common.add_argument("--solver", choices=("iterative", "ippe"), default="iterative")
    common.add_argument("--ransac", action="store_true", help="Reject pose outliers (requires >=6 corners)")
    common.add_argument("--refine", action="store_true", help="Refine candidate poses with Levenberg-Marquardt")
    common.add_argument("--ransac-threshold-px", type=_positive_float, default=3.)
    common.add_argument("--min-corners", type=_positive_int, default=6, help="Quality gate; must be >=4 (default: 6)")
    common.add_argument("--min-coverage", type=_fraction, default=.25)
    common.add_argument("--max-rmse-px", type=_positive_float, default=2.)
    common.add_argument("--max-error-px", type=_positive_float, default=5.)
    common.add_argument("--min-inlier-fraction", type=_fraction, default=.6)

    detect = commands.add_parser("detect", parents=[common], help="Detect in an image or select best camera frame")
    detect.add_argument("--board", required=True, help="Board name, e.g. front or tank01")
    detect.add_argument("--image", default=None)
    detect.add_argument("--frames", type=_positive_int, default=5, help="Sample N camera frames; retain best by corner count")
    detect.add_argument("--report", default=None)
    detect.add_argument("--save-image", default=None)

    monitor = argparse.ArgumentParser(add_help=False)
    monitor.add_argument("--output-dir", default="output", help="Parent directory for recordings and logs")
    monitor.add_argument("--save-frames", action="store_true", help="Save raw PNG frames (about 1.5 MB each)")
    monitor.add_argument("--reference-pose", help="Independent static reference pose JSON")
    monitor.add_argument("--rolling-seconds", type=_positive_float, default=2., help="Live jitter calculation window (default: 2 s)")
    monitor.add_argument("--window-seconds", type=_positive_float, default=30., help="Chart history (default: 30 s)")
    monitor.add_argument("--segment-minutes", type=_nonnegative_float, default=60.,
                         help="Start a new recording part after this many minutes; 0 disables (default: 60)")
    monitor.add_argument("--min-free-gb", type=_nonnegative_float, default=2.,
                         help="Stop recording when free disk space falls below this (default: 2 GB)")
    monitor.add_argument("--no-reconnect", dest="reconnect", action="store_false",
                         help="Stop instead of reconnecting after a camera failure")
    monitor.add_argument("--reconnect-max-seconds", type=_positive_float, default=30.,
                         help="Longest wait between reconnection attempts (default: 30 s)")
    monitor.add_argument("--hardware-reset-after", type=_nonnegative_int, default=3,
                         help="Hardware-reset the camera after every N failed reconnects; 0 disables (default: 3)")
    monitor.add_argument("--stall-seconds", type=_positive_float, default=30.,
                         help="Report the capture thread as stalled, and stop systemd watchdog pings, after this long (default: 30 s)")
    monitor.add_argument("--log-file", default=None,
                         help="Rotating log file (default: OUTPUT_DIR/logs/charuco-realsense.log; empty string disables)")
    monitor.set_defaults(frames=None, seconds=None)

    gui = commands.add_parser("gui", parents=[common, monitor], help="Desktop camera monitor with jitter charts and recording controls")
    gui.add_argument("--board", help="Board name (default: front if present, otherwise the first board)")
    gui.add_argument("--auto-record", action="store_true", help="Start recording as soon as the camera is ready")

    service = commands.add_parser("service", parents=[common, monitor],
                                  help="Headless 24/7 recorder: reconnects, rotates parts, logs, systemd watchdog")
    service.add_argument("--board", required=True, help="Board name, e.g. front or tank01")
    service.add_argument("--no-record", action="store_true", help="Monitor and log camera health without recording")

    for name in ("stream", "benchmark"):
        capture = commands.add_parser(name, parents=[common], help="Live GUI" if name == "stream" else "Record raw measurements and jitter statistics")
        capture.add_argument("--board", required=True, help="Board name, e.g. front or tank01")
        capture.add_argument("--frames", type=_positive_int, default=None, help="Stop after N recorded frames")
        capture.add_argument("--seconds", type=_positive_float, default=60. if name == "benchmark" else None)
        capture.add_argument("--reference-pose", help="Independent static ground-truth pose JSON")
        capture.add_argument("--save-frames", action="store_true", help="Also save unannotated PNG frames")
        if name == "benchmark":
            capture.add_argument("--output", required=True, help="New recording directory")
            capture.add_argument("--show", action="store_true", help="Also display live overlays")
        else:
            capture.add_argument("--record", help="New recording directory (automatically enables pose)")

    summary = commands.add_parser("summarize", help="Recompute statistics from a saved recording without hardware")
    summary.add_argument("--record", required=True)
    summary.add_argument("--reference-pose", help="Override/add independent static reference pose")
    summary.add_argument("--report", help="Also write the summary JSON here")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if hasattr(args, "manifest") and args.manifest is None:
            args.manifest = _default_manifest(args.generator_path)
        if args.command == "list":
            for device in list_devices():
                print(json.dumps(device, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "summarize":
            reference = load_reference_pose(args.reference_pose) if args.reference_pose else None
            summary = summarize_recording(args.record, reference)
            if args.report:
                write_json(args.report, summary)
            print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "detect":
            return _cmd_detect(args)
        if args.command == "gui":
            from .gui import run_gui
            return run_gui(args)
        if args.command == "service":
            from .service import run_service
            return run_service(args)
        return _cmd_capture(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except (CameraError, ValueError, KeyError, TypeError, OSError, cv2.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
