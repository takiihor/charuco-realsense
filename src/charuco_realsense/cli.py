"""Detect boards, display a live stream, and record repeatability experiments."""
import argparse
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


def _load_board(manifest_path: str, board_name: str, generator_path: str | None):
    if generator_path is not None:
        sys.path.insert(0, str(Path(generator_path) / "src"))
    try:
        from charuco_generator.board import reconstruct_board
        from charuco_generator.validation import load_manifest
    except ImportError as exc:
        raise ValueError("Install charuco-generator with --no-deps or supply --generator-path") from exc
    manifest = load_manifest(manifest_path)
    board = reconstruct_board(manifest, board_name)
    record = next((b for b in manifest["boards"] if b["name"] == board_name), None)
    if record is None:
        raise ValueError(f"Unknown board {board_name!r}")
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
    if pose and any(detection.marker_ids.count(i) > 1 for i in record["marker_ids"]):
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


def _draw(overlay: np.ndarray, detection: Detection) -> np.ndarray:
    for corner, marker_id in zip(detection.marker_corners, detection.marker_ids):
        pts = np.asarray(corner).reshape(4, 2).astype(int)
        cv2.polylines(overlay, [pts], True, (0, 0, 255), 2)
        cv2.putText(overlay, str(marker_id), (int(pts[0][0]), int(pts[0][1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    for corner, charuco_id in zip(detection.charuco_corners, detection.charuco_ids):
        x, y = (int(round(float(v))) for v in corner.ravel())
        cv2.circle(overlay, (x, y), 3, (0, 255, 0), -1)
        cv2.putText(overlay, str(charuco_id), (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return overlay


def _open_camera(args):
    return RealsenseCamera(serial=args.serial,
                           config=StreamConfig(args.width, args.height, args.fps, args.depth))


def _settle_camera(camera, args):
    deadline = time.monotonic() + args.warmup_seconds
    while time.monotonic() < deadline:
        camera.read()
    if args.lock_exposure:
        camera.lock_exposure()
        # Discard a few queued frames captured before the option took effect.
        for _ in range(3):
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
    common.add_argument("--manifest", default="../charuco-generator/output/board_manifest.json")
    common.add_argument("--board", required=True, help="Board name, e.g. front")
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
    detect.add_argument("--image", default=None)
    detect.add_argument("--frames", type=_positive_int, default=5, help="Sample N camera frames; retain best by corner count")
    detect.add_argument("--report", default=None)
    detect.add_argument("--save-image", default=None)

    gui = commands.add_parser("gui", parents=[common], help="Desktop camera monitor with jitter charts and recording controls")
    gui.add_argument("--output-dir", default="output", help="Parent directory for UI recordings")
    gui.add_argument("--save-frames", action="store_true", help="Initially enable saving raw PNG frames")
    gui.add_argument("--reference-pose", help="Independent static reference pose JSON")
    gui.add_argument("--rolling-seconds", type=_positive_float, default=2., help="Live jitter calculation window (default: 2 s)")
    gui.add_argument("--window-seconds", type=_positive_float, default=30., help="Chart history (default: 30 s)")
    gui.set_defaults(frames=None, seconds=None)

    for name in ("stream", "benchmark"):
        capture = commands.add_parser(name, parents=[common], help="Live GUI" if name == "stream" else "Record raw measurements and jitter statistics")
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
        return _cmd_capture(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except (CameraError, ValueError, KeyError, TypeError, OSError, cv2.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
