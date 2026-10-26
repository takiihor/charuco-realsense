import numpy as np
import pytest
from dataclasses import replace

import cv2

from charuco_generator.board import reconstruct_board
from charuco_realsense.detect import detect_charuco
from charuco_realsense.pose import PoseOptions, estimate_pose


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


@pytest.mark.parametrize("solver,refine", [("iterative", False), ("ippe", False), ("ippe", True)])
def test_known_pose_with_distortion(known_pose, solver, refine):
    board, detection, intrinsics, rvec, tvec = known_pose
    pose = estimate_pose(board, detection, intrinsics, PoseOptions(solver=solver, refine=refine))
    assert pose["quality_passed"]
    np.testing.assert_allclose(pose["tvec_m"], tvec, atol=1e-7)
    np.testing.assert_allclose(pose["rotation_matrix"], cv2.Rodrigues(rvec)[0], atol=1e-6)
    assert pose["reprojection_rmse_px"] < 1e-6
    assert pose["board_coverage_fraction"] == pytest.approx(1)
    assert pose["positive_depth"]
    assert pose["candidate_count"] == (2 if solver == "ippe" else 1)


def test_ransac_rejects_outliers_without_hiding_them(known_pose):
    board, detection, intrinsics, _, tvec = known_pose
    pixels = detection.charuco_corners.copy()
    pixels[:2, 0, :] += [40., -30.]
    detection = replace(detection, charuco_corners=pixels)
    cv2.setRNGSeed(42)
    pose = estimate_pose(board, detection, intrinsics, PoseOptions(ransac=True))
    assert pose is not None and pose["quality_passed"]
    assert pose["inlier_count"] == len(detection.charuco_ids) - 2
    np.testing.assert_allclose(pose["tvec_m"], tvec, atol=1e-6)
    assert pose["all_corner_reprojection_max_px"] > 40
    assert pose["reprojection_max_px"] < 1e-5
    assert sum(not c["inlier"] for c in pose["corner_errors"]) == 2


def test_quality_gates_keep_raw_pose(known_pose):
    board, detection, intrinsics, _, _ = known_pose
    pose = estimate_pose(board, detection, intrinsics, PoseOptions(min_corners=20))
    assert pose["ok"] and not pose["quality_passed"]
    assert "insufficient_corners" in pose["quality_reasons"]
    pixels = detection.charuco_corners.copy()
    pixels[0, 0, :] += [30., -20.]
    pose = estimate_pose(board, replace(detection, charuco_corners=pixels), intrinsics)
    assert not pose["quality_passed"]
    assert "high_reprojection_rmse" in pose["quality_reasons"]


def test_collinear_and_invalid_ids(known_pose):
    board, detection, intrinsics, _, _ = known_pose
    line = replace(detection, charuco_ids=detection.charuco_ids[:4],
                   charuco_corners=detection.charuco_corners[:4])
    assert estimate_pose(board, line, intrinsics) is None
    with pytest.raises(ValueError, match="unique valid"):
        estimate_pose(board, replace(detection, charuco_ids=(0,) * len(detection.charuco_ids)), intrinsics)


def test_behind_camera_pose_fails_quality(known_pose, monkeypatch):
    board, detection, intrinsics, _, _ = known_pose
    monkeypatch.setattr(cv2, "solvePnP", lambda *a, **k: (True, np.zeros((3, 1)), np.array([[0.], [0.], [-1.]])))
    pose = estimate_pose(board, detection, intrinsics)
    assert not pose["positive_depth"]
    assert "nonpositive_depth" in pose["quality_reasons"]


def test_print_scale_bias_can_have_zero_reprojection_error(known_pose):
    from charuco_realsense.calibration import opencv_calibration

    board, detection, intrinsics, rvec, tvec = known_pose
    matrix, coeffs = opencv_calibration(intrinsics)
    pixels, _ = cv2.projectPoints(np.asarray(board.getChessboardCorners(), float) * 1.01,
                                 rvec, tvec, matrix, coeffs)
    pose = estimate_pose(board, replace(detection, charuco_corners=pixels), intrinsics)
    assert pose["reprojection_rmse_px"] < 1e-6
    np.testing.assert_allclose(pose["tvec_m"], tvec / 1.01, atol=1e-7)


def test_small_corner_cluster_fails_coverage_gate(known_pose):
    board, detection, intrinsics, _, _ = known_pose
    indices = [0, 1, 4, 5]
    partial = replace(detection, charuco_ids=tuple(indices), charuco_corners=detection.charuco_corners[indices])
    pose = estimate_pose(board, partial, intrinsics, PoseOptions(min_corners=4, min_coverage=.25))
    assert pose is not None and pose["ok"]
    assert pose["board_coverage_fraction"] == pytest.approx(1 / 6)
    assert pose["quality_reasons"] == ["insufficient_board_coverage"]
