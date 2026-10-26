"""Command line interface: list devices, detect on demand, stream with overlays."""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .detect import Detection, detect_charuco, report_for_board
from .pose import estimate_pose


def _load_board(manifest_path: str, board_name: str, generator_path: str | None):
    """Reconstruct a validated board from a charuco-generator manifest."""
    if generator_path is not None:
        sys.path.insert(0, str(Path(generator_path) / "src"))
    try:
        from charuco_generator.board import reconstruct_board
        from charuco_generator.validation import load_manifest
    except ImportError as exc:
        raise SystemExit(
            "charuco-generator is not importable. Install it into this environment "
            "(pip install -e ../charuco-generator --no-deps) or pass --generator-path "
            "/path/to/charuco-generator.") from exc
    manifest = load_manifest(manifest_path)
    board = reconstruct_board(manifest, board_name)
    record = next((b for b in manifest["boards"] if b["name"] == board_name), None)
    if record is None:
        raise SystemExit(f"Unknown board {board_name!r}; manifest has "
                         f"{[b['name'] for b in manifest['boards']]}")
    return board, record, manifest


def _detect_image(board, record, image_bgr: np.ndarray, border_bits: int, include_pose: bool, intrinsics: dict | None) -> dict:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    detection = detect_charuco(board, gray, border_bits)
    report = {
        "board": record["name"],
        "identifier": record["identifier"],
        "image_pixels": list(detection.image_pixels),
        **report_for_board(detection, record["marker_ids"], record["expected_charuco_corner_count"]),
    }
    if include_pose and intrinsics is not None:
        report["pose"] = estimate_pose(board, detection, intrinsics)
    report["passed"] = bool(report["matched_marker_ids"])
    return report


def _write_report(path: str, report: dict) -> None:
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                          encoding="utf-8")


def _draw(overlay: np.ndarray, detection: Detection) -> np.ndarray:
    for corner, marker_id in zip(detection.marker_corners, detection.marker_ids):
        pts = np.asarray(corner, dtype=np.float64).reshape(4, 2).astype(int)
        cv2.polylines(overlay, [pts], True, (0, 0, 255), 2)
        cv2.putText(overlay, str(marker_id), (int(pts[0][0]), int(pts[0][1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    for corner, charuco_id in zip(detection.charuco_corners, detection.charuco_ids):
        x, y = (int(round(float(v))) for v in corner.ravel())
        cv2.circle(overlay, (x, y), 3, (0, 255, 0), -1)
        cv2.putText(overlay, str(charuco_id), (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return overlay


def _cmd_list(args) -> int:
    from .camera import list_devices
    for device in list_devices():
        print(json.dumps(device, sort_keys=True))
    return 0


def _cmd_detect(args) -> int:
    board, record, _ = _load_board(args.manifest, args.board, args.generator_path)
    border_bits = record["geometry"]["border_bits"]
    if args.image:
        image = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if image is None:
            print(f"error: cannot read image {args.image}", file=sys.stderr)
            return 2
        report = _detect_image(board, record, image, border_bits, args.pose, None)
        report["source"] = args.image
    else:
        from .camera import CameraError, RealsenseCamera, StreamConfig
        try:
            with RealsenseCamera(serial=args.serial,
                                 config=StreamConfig(args.width, args.height, args.fps)) as camera:
                best = None
                for _ in range(args.frames):
                    frame = camera.read()
                    candidate = _detect_image(board, record, frame.bgr, border_bits,
                                              args.pose, frame.intrinsics)
                    if best is None or candidate["detected_charuco_corner_count"] > best["detected_charuco_corner_count"]:
                        best = candidate
                report = best
            report["source"] = "realsense"
            report["frames_sampled"] = args.frames
        except CameraError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    if args.save_image:
        cv2.imwrite(args.save_image, image)
    if args.report:
        _write_report(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 1


def _cmd_stream(args) -> int:
    from .camera import CameraError, RealsenseCamera, StreamConfig
    board, record, _ = _load_board(args.manifest, args.board, args.generator_path)
    border_bits = record["geometry"]["border_bits"]
    try:
        with RealsenseCamera(serial=args.serial,
                             config=StreamConfig(args.width, args.height, args.fps)) as camera:
            print(f"Streaming {record['name']} ({record['identifier']}); press q to quit")
            fps = 0.0
            last = time.monotonic()
            while True:
                frame = camera.read()
                now = time.monotonic()
                fps = 0.7 * fps + 0.3 * (1.0 / max(now - last, 1e-6))
                last = now
                detection = detect_charuco(board, cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2GRAY),
                                           border_bits)
                overlay = _draw(frame.bgr.copy(), detection)
                pose = estimate_pose(board, detection, frame.intrinsics) if args.pose else None
                status = (f"{record['name']}: {len(detection.marker_ids)} markers, "
                          f"{len(detection.charuco_ids)} corners, {fps:.0f} fps")
                if pose:
                    x, y, z = pose["position_mm"]
                    status += f" | z={z:.1f} mm"
                cv2.putText(overlay, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0) if detection.found else (0, 0, 255), 2)
                cv2.imshow("charuco-realsense", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except CameraError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        cv2.destroyAllWindows()
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Detect charuco-generator ChArUco boards with an Intel RealSense D435i")
    root.add_argument("--version", action="version", version=f"charuco-realsense {__version__}")
    commands = root.add_subparsers(dest="command", required=True)

    commands.add_parser("list", help="List connected RealSense devices")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", default="../charuco-generator/output/board_manifest.json",
                        help="Generator manifest (default: sibling checkout)")
    common.add_argument("--board", required=True, help="Board name from the manifest, e.g. front")
    common.add_argument("--generator-path", default=None,
                        help="charuco-generator checkout to import when not installed")

    det = commands.add_parser("detect", parents=[common],
                              help="Detect once (camera) or in an image file")
    det.add_argument("--image", default=None, help="Offline mode: read a PNG/JPG instead of the camera")
    det.add_argument("--frames", type=int, default=5, help="Camera frames to sample; best by corner count")
    det.add_argument("--pose", action="store_true", help="Estimate board pose with solvePnP (camera mode)")
    det.add_argument("--serial", default=None, help="RealSense serial number")
    det.add_argument("--width", type=int, default=1280)
    det.add_argument("--height", type=int, default=720)
    det.add_argument("--fps", type=int, default=30)
    det.add_argument("--report", default=None, help="Also save the JSON report here")
    det.add_argument("--save-image", default=None, help="Also save the analyzed frame here")

    stream = commands.add_parser("stream", parents=[common],
                                 help="Live view with marker/corner overlays (requires GUI)")
    stream.add_argument("--pose", action="store_true", help="Show estimated z distance")
    stream.add_argument("--serial", default=None)
    stream.add_argument("--width", type=int, default=1280)
    stream.add_argument("--height", type=int, default=720)
    stream.add_argument("--fps", type=int, default=30)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "list":
            return _cmd_list(args)
        if args.command == "detect":
            return _cmd_detect(args)
        return _cmd_stream(args)
    except (ValueError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
