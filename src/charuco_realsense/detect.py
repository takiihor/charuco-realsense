"""ChArUco detection from grayscale images, mirroring charuco-generator's validation."""
from dataclasses import dataclass

import cv2
import numpy as np


class DetectionError(ValueError):
    pass


@dataclass(frozen=True)
class Detection:
    """Detected ArUco markers and internal ChArUco corners in one image."""

    image_pixels: tuple[int, int]  # (width, height)
    marker_ids: tuple[int, ...]
    marker_corners: np.ndarray  # (N, 4, 2) float32, px
    charuco_ids: tuple[int, ...]
    charuco_corners: np.ndarray  # (M, 1, 2) float32, px
    rejected_candidates: int

    @property
    def found(self) -> bool:
        return len(self.marker_ids) > 0


def detect_charuco(board, image_gray: np.ndarray, border_bits: int = 1) -> Detection:
    """Detect markers and internal corners using the modern OpenCV API.

    ``board`` is a cv2.aruco.CharucoBoard (e.g. from charuco_generator.reconstruct_board).
    ``border_bits`` must match the board's configured marker border.
    """
    if image_gray.ndim != 2:
        raise DetectionError("image_gray must be a single-channel image")
    params = cv2.aruco.DetectorParameters()
    params.markerBorderBits = border_bits
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    marker_corners, marker_ids, rejected = cv2.aruco.ArucoDetector(
        board.getDictionary(), params).detectMarkers(image_gray)
    charuco_params = cv2.aruco.CharucoParameters()
    charuco_params.tryRefineMarkers = False
    charuco_corners, charuco_ids, _, _ = cv2.aruco.CharucoDetector(
        board, charuco_params, params).detectBoard(image_gray)
    if marker_ids is None:
        marker_ids = np.asarray([], dtype=np.int32)
    if marker_corners is None:
        marker_corners = np.zeros((0, 4, 2), dtype=np.float32)
    if charuco_ids is None:
        charuco_ids = np.asarray([], dtype=np.int32)
    if charuco_corners is None:
        charuco_corners = np.zeros((0, 1, 2), dtype=np.float32)
    return Detection(
        image_pixels=(image_gray.shape[1], image_gray.shape[0]),
        marker_ids=tuple(int(v) for v in np.asarray(marker_ids).ravel()),
        marker_corners=np.asarray(marker_corners, dtype=np.float32),
        charuco_ids=tuple(int(v) for v in np.asarray(charuco_ids).ravel()),
        charuco_corners=np.asarray(charuco_corners, dtype=np.float32),
        rejected_candidates=len(rejected) if rejected is not None else 0,
    )


def report_for_board(detection: Detection, expected_ids, expected_corner_count: int) -> dict:
    """Split detected IDs into matched / missing / foreign for a known board."""
    expected = list(expected_ids)
    detected = detection.marker_ids
    return {
        "expected_marker_ids": expected,
        "expected_charuco_corner_count": expected_corner_count,
        "detected_marker_ids": sorted(detected),
        "detected_marker_count": len(detected),
        "matched_marker_ids": sorted(set(detected) & set(expected)),
        "missing_marker_ids": sorted(set(expected) - set(detected)),
        "foreign_marker_ids": sorted(set(detected) - set(expected)),
        "duplicate_marker_ids": sorted({v for v in detected if detected.count(v) > 1}),
        "detected_charuco_ids": sorted(detection.charuco_ids),
        "detected_charuco_corner_count": len(detection.charuco_ids),
        "rejected_marker_candidate_count": detection.rejected_candidates,
        "full_expected_visibility": (
            sorted(detected) == sorted(expected)
            and sorted(detection.charuco_ids) == list(range(expected_corner_count))),
    }
