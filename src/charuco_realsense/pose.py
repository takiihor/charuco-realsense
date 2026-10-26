"""Board pose, residuals, planar ambiguity, and configurable quality checks."""
from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import opencv_calibration
from .detect import Detection


class PoseInputError(ValueError):
    """Invalid image correspondences, rather than an invalid calibration."""


@dataclass(frozen=True)
class PoseOptions:
    solver: str = "iterative"
    ransac: bool = False
    refine: bool = False
    ransac_threshold_px: float = 3.0
    min_corners: int = 6
    min_coverage: float = 0.25
    max_rmse_px: float = 2.0
    max_error_px: float = 5.0
    min_inlier_fraction: float = 0.6

    def __post_init__(self):
        if self.solver not in {"iterative", "ippe"}:
            raise ValueError("Pose solver must be iterative or ippe")
        if not isinstance(self.min_corners, int) or self.min_corners < 4:
            raise ValueError("min_corners must be an integer >= 4")
        for name in ("ransac_threshold_px", "max_rmse_px", "max_error_px"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("min_coverage", "min_inlier_fraction"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")


def _hull_area(points):
    return float(cv2.contourArea(cv2.convexHull(np.asarray(points, np.float32).reshape(-1, 2))))


def _residuals(points, pixels, matrix, coeffs, rvec, tvec):
    projected, _ = cv2.projectPoints(points, rvec, tvec, matrix, coeffs)
    return np.linalg.norm(projected.reshape(-1, 2) - pixels.reshape(-1, 2), axis=1)


def estimate_pose(board, detection: Detection, intrinsics: dict,
                  options: PoseOptions | None = None) -> dict | None:
    """Return an unfiltered pose and quality diagnostics, or None if unsolvable.

    Board coordinates use metres, X right, Y down, Z into the board. Quality
    thresholds are operational choices, not physical accuracy guarantees. With
    RANSAC, quality residuals/coverage use inliers; all observations are retained.
    """
    options = options or PoseOptions()
    matrix, coeffs = opencv_calibration(intrinsics, detection.image_pixels)
    ids = np.asarray(detection.charuco_ids, dtype=np.int32)
    all_points = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    image_points = np.asarray(detection.charuco_corners, dtype=np.float64).reshape(-1, 2)
    if (len(ids) != len(image_points) or len(np.unique(ids)) != len(ids)
            or np.any(ids < 0) or np.any(ids >= len(all_points))
            or not np.isfinite(image_points).all()):
        raise PoseInputError("Pose requires unique valid corner IDs and matching finite image points")
    if len(ids) < 4 or board.checkCharucoCornersCollinear(ids):
        return None
    selected = np.ascontiguousarray(all_points[ids])
    image_points = np.ascontiguousarray(image_points)
    inliers = np.arange(len(ids))
    try:
        if options.ransac:
            # Robust fitting cannot identify outliers from a minimal four-point set.
            if len(ids) < 6:
                return None
            ok, _, _, indices = cv2.solvePnPRansac(
                selected, image_points, matrix, coeffs, iterationsCount=200,
                reprojectionError=options.ransac_threshold_px, confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP)
            if not ok or indices is None or len(indices) < 4:
                return None
            inliers = np.sort(indices.ravel())
            if board.checkCharucoCornersCollinear(ids[inliers]):
                return None
        fit_points = np.ascontiguousarray(selected[inliers])
        fit_pixels = np.ascontiguousarray(image_points[inliers])
        if options.solver == "ippe":
            count, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                fit_points, fit_pixels, matrix, coeffs, flags=cv2.SOLVEPNP_IPPE)
            if not count:
                return None
        else:
            ok, rvec, tvec = cv2.solvePnP(
                fit_points, fit_pixels, matrix, coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return None
            rvecs, tvecs = (rvec,), (tvec,)
        candidates = []
        for rvec, tvec in zip(rvecs, tvecs):
            rvec, tvec = np.asarray(rvec, float).reshape(3, 1), np.asarray(tvec, float).reshape(3, 1)
            if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
                continue
            if options.refine:
                rvec, tvec = cv2.solvePnPRefineLM(fit_points, fit_pixels, matrix, coeffs, rvec, tvec)
            rotation, _ = cv2.Rodrigues(rvec)
            errors = _residuals(selected, image_points, matrix, coeffs, rvec, tvec)
            if not np.isfinite(errors).all():
                continue
            positive = bool(np.all((all_points @ rotation.T + tvec.ravel())[:, 2] > 0))
            candidates.append({"rvec": rvec.ravel(), "tvec": tvec.ravel(), "rotation": rotation,
                               "errors": errors, "positive": positive,
                               "rmse": float(np.sqrt(np.mean(errors[inliers] ** 2)))})
    except cv2.error:
        # Degenerate numerical configurations must not terminate a recording.
        return None
    if not candidates:
        return None
    candidates.sort(key=lambda c: (not c["positive"], c["rmse"]))
    best = candidates[0]
    errors = best["errors"]
    coverage = _hull_area(fit_points[:, :2]) / max(_hull_area(all_points[:, :2]), 1e-15)
    image_coverage = _hull_area(fit_pixels) / float(np.prod(detection.image_pixels))
    max_error = float(errors[inliers].max())
    reasons = []
    if not best["positive"]:
        reasons.append("nonpositive_depth")
    if len(inliers) < options.min_corners:
        reasons.append("insufficient_corners")
    if coverage < options.min_coverage:
        reasons.append("insufficient_board_coverage")
    if len(inliers) / len(ids) < options.min_inlier_fraction:
        reasons.append("insufficient_inlier_fraction")
    if best["rmse"] > options.max_rmse_px:
        reasons.append("high_reprojection_rmse")
    if max_error > options.max_error_px:
        reasons.append("high_corner_error")
    inlier_set = set(inliers.tolist())
    return {
        "ok": True,
        "quality_passed": not reasons,
        "quality_reasons": reasons,
        "solver": options.solver,
        "ransac": options.ransac,
        "refined": options.refine,
        "corner_count": len(ids),
        "inlier_count": len(inliers),
        "inlier_fraction": len(inliers) / len(ids),
        "inlier_charuco_ids": ids[inliers].tolist(),
        "positive_depth": best["positive"],
        "board_coverage_fraction": float(coverage),
        "image_coverage_fraction": float(image_coverage),
        "reprojection_rmse_px": best["rmse"],
        "reprojection_max_px": max_error,
        "all_corner_reprojection_rmse_px": float(np.sqrt(np.mean(errors ** 2))),
        "all_corner_reprojection_max_px": float(errors.max()),
        "corner_errors": [{"id": int(i), "error_px": float(e), "inlier": j in inlier_set}
                          for j, (i, e) in enumerate(zip(ids, errors))],
        "candidate_count": len(candidates),
        "candidate_rmse_gap_px": (abs(candidates[1]["rmse"] - best["rmse"])
                                  if len(candidates) > 1 else None),
        "candidates": [{"rvec": c["rvec"].tolist(), "tvec_m": c["tvec"].tolist(),
                        "reprojection_rmse_px": c["rmse"], "positive_depth": c["positive"]}
                       for c in candidates],
        "rotation_matrix": best["rotation"].tolist(),
        "rvec": best["rvec"].tolist(),
        "tvec_m": best["tvec"].tolist(),
        "position_mm": (best["tvec"] * 1000.0).tolist(),
        "board_origin": "outer top-left pattern corner; X right, Y down, Z into board (plane Z=0)",
        "units": "rvec radians; tvec_m metres; position_mm millimetres",
        "note": "Image fit and repeatability do not establish physical accuracy",
    }
