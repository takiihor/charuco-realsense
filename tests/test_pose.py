import numpy as np

import cv2

from charuco_generator.board import reconstruct_board
from charuco_realsense.detect import detect_charuco
from charuco_realsense.pose import estimate_pose


def _simulated_photo(png_path: str, intrinsics: dict) -> np.ndarray:
    """Mild perspective warp at simulated intrinsics, mirroring the generator's photo test."""
    image = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, None, fx=0.45, fy=0.45, interpolation=cv2.INTER_AREA)
    h, w = image.shape
    source = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    target = np.float32([[35, 15], [w - 45, 50], [w - 12, h - 30], [10, h - 12]])
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(source, target), (w, h),
                               borderValue=255)


def test_pose_on_simulated_photo(artifacts):
    directory, manifest = artifacts
    record = manifest["boards"][0]
    board = reconstruct_board(manifest, record["name"])
    h, w = 720, 1280
    intrinsics = {"fx": 1000.0, "fy": 1000.0, "ppx": w / 2, "ppy": h / 2}
    photo = _simulated_photo(str(directory / record["files"]["png"]), intrinsics)
    detection = detect_charuco(board, photo, record["geometry"]["border_bits"])
    assert detection.found
    pose = estimate_pose(board, detection, intrinsics)
    assert pose is not None
    rotation = np.asarray(pose["rotation_matrix"])
    # Near-frontal view: board axes align with camera axes (X right, Y down, Z into board).
    assert float(np.linalg.norm(rotation - np.eye(3))) < 0.3
    # Pattern is ~100 mm wide spanning ~1000 px at fx=1000 -> board ~0.1 m away.
    z = pose["tvec_m"][2]
    assert 0.03 < z < 0.3


def test_pose_requires_four_corners(artifacts):
    directory, manifest = artifacts
    record = manifest["boards"][0]
    board = reconstruct_board(manifest, record["name"])
    blank = np.full((720, 1280), 255, dtype=np.uint8)
    detection = detect_charuco(board, blank, record["geometry"]["border_bits"])
    assert estimate_pose(board, detection,
                         {"fx": 1000.0, "fy": 1000.0, "ppx": 640.0, "ppy": 360.0}) is None
