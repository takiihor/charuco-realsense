"""ChArUco detection from grayscale images, mirroring charuco-generator's validation."""
from collections import Counter
from dataclasses import dataclass
import math

import cv2
import numpy as np

from .calibration import opencv_calibration


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


def refine_corners(image_gray: np.ndarray, pixels: np.ndarray) -> np.ndarray | None:
    """Refine chessboard intersections in OpenCV's pixel-centre coordinates.

    OpenCV 4.11's detectBoard adds (0.5, 0.5) after its internal cornerSubPix
    pass (opencv/opencv#25539). Re-refining on the original image removes this
    bias without a version-dependent subtraction. Reject a result that moves
    any corner more than one pixel instead of accepting an unrelated feature.
    The returned array is (N, 2); None means the frame cannot be refined.
    """
    points = np.asarray(pixels, np.float32).reshape(-1, 1, 2)
    if len(points) < 2 or not np.isfinite(points).all():
        return None
    spacing = math.sqrt(max(cv2.contourArea(cv2.convexHull(points)), 1.0) / len(points))
    half = int(min(8, max(2, spacing * 0.25)))
    refined = cv2.cornerSubPix(image_gray, points.copy(), (half, half), (-1, -1),
                               (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-4))
    if not np.isfinite(refined).all() or np.max(np.linalg.norm((refined - points).reshape(-1, 2), axis=1)) > 1.0:
        return None
    return refined.reshape(-1, 2)


class CharucoDetector:
    """Reusable detectors; one marker pass per frame, including blank frames."""

    def __init__(self, board, border_bits: int = 1, *, refine: bool = True):
        if not isinstance(border_bits, int) or border_bits < 1:
            raise DetectionError("border_bits must be a positive integer")
        self.settings = {"border_bits": border_bits, "marker_corner_refinement": "none",
                         "try_refine_markers": False, "charuco_corner_refinement": refine}
        self._refine = refine
        params = cv2.aruco.DetectorParameters()
        params.markerBorderBits = border_bits
        # Keep marker refinement off; refine only chessboard intersections below.
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        self._markers = cv2.aruco.ArucoDetector(board.getDictionary(), params)
        self._charuco = cv2.aruco.CharucoDetector(board, cv2.aruco.CharucoParameters(), params)
        self._calibration_key = None

    def detect(self, image_gray: np.ndarray, intrinsics: dict | None = None) -> Detection:
        if image_gray.ndim != 2 or image_gray.dtype != np.uint8 or not image_gray.size:
            raise DetectionError("image_gray must be a nonempty uint8 single-channel image")
        size = (image_gray.shape[1], image_gray.shape[0])
        matrix, coeffs = (None, None) if intrinsics is None else opencv_calibration(intrinsics, size)
        key = () if matrix is None else (tuple(matrix.ravel()), tuple(coeffs))
        if key != self._calibration_key:
            params = cv2.aruco.CharucoParameters()
            params.tryRefineMarkers = False
            if matrix is not None:
                params.cameraMatrix = matrix
                params.distCoeffs = coeffs
            self._charuco.setCharucoParameters(params)
            self._calibration_key = key
        marker_corners, marker_ids, rejected = self._markers.detectMarkers(image_gray)
        charuco_corners, charuco_ids = None, None
        # Empty marker inputs cause OpenCV to run marker detection again; skip that.
        if marker_ids is not None and len(marker_ids):
            charuco_corners, charuco_ids, _, _ = self._charuco.detectBoard(
                image_gray, markerCorners=marker_corners, markerIds=marker_ids)
            if self._refine and charuco_corners is not None and len(charuco_corners):
                refined = refine_corners(image_gray, charuco_corners)
                if refined is None:
                    charuco_corners, charuco_ids = None, None
                else:
                    charuco_corners = refined.reshape(-1, 1, 2)
        return Detection(
            image_pixels=size,
            marker_ids=tuple(int(v) for v in np.asarray(marker_ids).ravel()) if marker_ids is not None else (),
            marker_corners=(np.asarray(marker_corners, dtype=np.float32).reshape(-1, 4, 2)
                            if marker_corners is not None else np.empty((0, 4, 2), np.float32)),
            charuco_ids=tuple(int(v) for v in np.asarray(charuco_ids).ravel()) if charuco_ids is not None else (),
            charuco_corners=(np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 1, 2)
                             if charuco_corners is not None else np.empty((0, 1, 2), np.float32)),
            rejected_candidates=len(rejected) if rejected is not None else 0,
        )


def detect_charuco(board, image_gray: np.ndarray, border_bits: int = 1,
                   intrinsics: dict | None = None) -> Detection:
    """Convenience API for one image; reuse CharucoDetector for sequences."""
    return CharucoDetector(board, border_bits).detect(image_gray, intrinsics)


def report_for_board(detection: Detection, expected_ids, expected_corner_count: int) -> dict:
    """Split detected IDs into matched / missing / foreign for a known board."""
    expected = list(expected_ids)
    detected = detection.marker_ids
    marker_counts = Counter(detected)
    corner_counts = Counter(detection.charuco_ids)
    return {
        "expected_marker_ids": expected,
        "expected_charuco_corner_count": expected_corner_count,
        "detected_marker_ids": sorted(detected),
        "detected_marker_count": len(detected),
        "matched_marker_ids": sorted(set(detected) & set(expected)),
        "missing_marker_ids": sorted(set(expected) - set(detected)),
        "foreign_marker_ids": sorted(set(detected) - set(expected)),
        "duplicate_marker_ids": sorted(i for i, count in marker_counts.items() if count > 1),
        "detected_charuco_ids": sorted(detection.charuco_ids),
        "detected_charuco_corner_count": len(detection.charuco_ids),
        "duplicate_charuco_ids": sorted(i for i, count in corner_counts.items() if count > 1),
        # Explicit ID/coordinate pairs avoid losing association when IDs are sorted.
        "charuco_corners": [{"id": int(i), "pixel": p.ravel().tolist()}
                            for i, p in zip(detection.charuco_ids, detection.charuco_corners)],
        "marker_corners": [{"id": int(i), "pixels": p.tolist()}
                           for i, p in zip(detection.marker_ids, detection.marker_corners)],
        "rejected_marker_candidate_count": detection.rejected_candidates,
        "full_expected_visibility": (
            sorted(detected) == sorted(expected)
            and sorted(detection.charuco_ids) == list(range(expected_corner_count))),
    }
