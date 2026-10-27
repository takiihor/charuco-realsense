"""Repeatability and reference-pose accuracy statistics for raw frame records."""
from array import array
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np


def validate_reference_pose(reference: dict) -> dict:
    if not isinstance(reference, dict):
        raise ValueError("Reference pose must be a JSON object")
    translation = np.asarray(reference["tvec_m"], float)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("Reference tvec_m must contain three finite values in metres")
    if "rotation_matrix" in reference:
        rotation = np.asarray(reference["rotation_matrix"], float)
    else:
        rvec = np.asarray(reference["rvec"], float)
        if rvec.shape != (3,) or not np.isfinite(rvec).all():
            raise ValueError("Reference rvec must contain three finite values in radians")
        rotation, _ = cv2.Rodrigues(rvec)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6, rtol=0)):
        raise ValueError("Reference rotation_matrix must be a proper 3x3 rotation")
    return {"tvec_m": translation.tolist(), "rotation_matrix": rotation.tolist()}


def load_reference_pose(path: str) -> dict:
    return validate_reference_pose(json.loads(Path(path).read_text(encoding="utf-8")))


def rotation_angle_deg(rotation, reference) -> float:
    relative = np.asarray(rotation) @ np.asarray(reference).T
    return float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.) / 2., -1., 1.))))


def pose_error(pose: dict, reference: dict) -> dict:
    delta = (np.asarray(pose["tvec_m"]) - np.asarray(reference["tvec_m"])) * 1000.
    return {"translation_error_mm": delta.tolist(),
            "translation_error_norm_mm": float(np.linalg.norm(delta)),
            "rotation_error_deg": rotation_angle_deg(pose["rotation_matrix"], reference["rotation_matrix"])}


def _distribution(values):
    if not len(values):
        return None
    values = np.asarray(values, float)
    return {"count": len(values), "mean": float(values.mean()),
            "rms": float(np.sqrt(np.mean(values ** 2))),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def rotation_angles_deg(rotations, reference) -> np.ndarray:
    """Vectorized rotation_angle_deg for an (N, 3, 3) stack; fast for long sessions."""
    rotations = np.asarray(rotations, float).reshape(-1, 3, 3)
    cosine = (np.einsum("nij,ij->n", rotations, np.asarray(reference, float)) - 1.) / 2.
    return np.degrees(np.arccos(np.clip(cosine, -1., 1.)))


def _mean_rotation(rotations):
    u, _, vh = np.linalg.svd(np.asarray(rotations).sum(axis=0))
    correction = np.diag([1., 1., np.linalg.det(u @ vh)])
    return u @ correction @ vh


class _Moments:
    """Online vector moments for per-ID corners, independent of observation order."""

    def __init__(self):
        self.count = 0
        self.mean = np.zeros(2)
        self.m2 = np.zeros(2)

    def add(self, value):
        value = np.asarray(value, float)
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def summary(self, frames):
        return {"samples": self.count,
                "missing_rate": 1. - self.count / frames if frames else None,
                "mean_pixel": self.mean.tolist() if self.count else None,
                "std_xy_px": np.sqrt(np.maximum(self.m2, 0) / (self.count - 1)).tolist() if self.count > 1 else None,
                "radial_rms_px": float(np.sqrt(max(float(self.m2.sum()), 0) / self.count)) if self.count > 1 else None}


class _PoseSeries:
    """Flat float64 buffers: about 130 bytes per pose instead of ~800 for small arrays."""

    def __init__(self):
        self.positions = array("d")
        self.rotations = array("d")
        self.rmse = array("d")
        self.all_rmse = array("d")
        self.coverage = array("d")
        self.corner_counts = array("d")

    def add(self, pose):
        position = np.asarray(pose["tvec_m"], float).reshape(3) * 1000.
        rotation = np.asarray(pose["rotation_matrix"], float).reshape(9)
        self.positions.extend(position.tolist())
        self.rotations.extend(rotation.tolist())
        self.rmse.append(pose["reprojection_rmse_px"])
        self.all_rmse.append(pose["all_corner_reprojection_rmse_px"])
        self.coverage.append(pose["board_coverage_fraction"])
        self.corner_counts.append(pose["inlier_count"])

    def summary(self, reference):
        n = len(self.rmse)
        result = {"samples": n, "position_jitter": None, "rotation_jitter": None,
                  "reprojection_rmse_px": _distribution(self.rmse),
                  "all_corner_reprojection_rmse_px": _distribution(self.all_rmse),
                  "board_coverage_fraction": _distribution(self.coverage),
                  "inlier_corner_count": _distribution(self.corner_counts), "accuracy": None}
        if not n:
            return result
        positions = np.array(self.positions, float).reshape(-1, 3)
        rotations = np.array(self.rotations, float).reshape(-1, 3, 3)
        mean = positions.mean(axis=0)
        mean_rotation = _mean_rotation(rotations)
        window = max(1, min(30, n // 10))
        displacement = np.linalg.norm(positions - mean, axis=1)
        angles = rotation_angles_deg(rotations, mean_rotation)
        result["position_jitter"] = {
            "mean_mm": mean.tolist(),
            "std_xyz_mm": positions.std(axis=0, ddof=1).tolist() if n > 1 else None,
            "rms_displacement_mm": float(np.sqrt(np.mean(displacement ** 2))) if n > 1 else None,
            "p95_displacement_mm": float(np.percentile(displacement, 95)) if n > 1 else None,
            "peak_to_peak_xyz_mm": np.ptp(positions, axis=0).tolist() if n > 1 else None,
            "drift_xyz_mm": (positions[-window:].mean(axis=0) - positions[:window].mean(axis=0)).tolist() if n > 1 else None,
            "drift_window_samples": window,
        }
        result["rotation_jitter"] = {
            "mean_rotation_matrix": mean_rotation.tolist(),
            "angular_deviation_deg": _distribution(angles) if n > 1 else None,
            "drift_deg": rotation_angle_deg(_mean_rotation(rotations[-window:]),
                                               _mean_rotation(rotations[:window])) if n > 1 else None,
        }
        if reference is not None:
            delta = positions - np.asarray(reference["tvec_m"]) * 1000.
            bias = delta.mean(axis=0)
            result["accuracy"] = {
                "translation_bias_xyz_mm": bias.tolist(),
                "translation_bias_norm_mm": float(np.linalg.norm(bias)),
                "translation_error_norm_mm": _distribution(np.linalg.norm(delta, axis=1)),
                "rotation_error_deg": _distribution(rotation_angles_deg(rotations, reference["rotation_matrix"])),
            }
        return result


class SessionStatistics:
    """Accumulate raw and accepted poses separately; failures stay in denominators.

    Pose arrays are retained for exact percentiles and rotation statistics; image
    coordinates use online moments. Images themselves are never retained here.
    """

    def __init__(self, expected_corner_count: int, reference_pose: dict | None = None):
        self.reference = validate_reference_pose(reference_pose) if reference_pose is not None else None
        self.corners = {i: _Moments() for i in range(expected_corner_count)}
        self.frames = 0
        self.counts = dict.fromkeys(("board_recognized", "pose_available", "pose_quality_passed", "full_expected_visibility"), 0)
        self.current_dropouts = dict.fromkeys(self.counts, 0)
        self.longest_dropouts = dict.fromkeys(self.counts, 0)
        self.all_poses = _PoseSeries()
        self.accepted_poses = _PoseSeries()
        self.processing_times = array("d")
        self.intervals = array("d")
        self.first_time = self.last_time = None
        self.last_frame_number = None
        self.last_timestamp = None
        self.last_domain = None
        self.frame_gaps = self.duplicate_frames = self.frame_number_resets = self.timestamp_resets = 0
        self.foreign_frames = self.duplicate_id_frames = 0
        self.missing_markers = {}
        self.quality_reasons = {}

    def add(self, record: dict):
        self.frames += 1
        for key in self.counts:
            present = bool(record[key])
            self.counts[key] += int(present)
            self.current_dropouts[key] = 0 if present else self.current_dropouts[key] + 1
            self.longest_dropouts[key] = max(self.longest_dropouts[key], self.current_dropouts[key])
        self.foreign_frames += bool(record["foreign_marker_ids"])
        self.duplicate_id_frames += bool(record["duplicate_marker_ids"])
        for marker_id in record["missing_marker_ids"]:
            key = str(marker_id)
            self.missing_markers[key] = self.missing_markers.get(key, 0) + 1
        for reason in record["pose_quality_reasons"]:
            self.quality_reasons[reason] = self.quality_reasons.get(reason, 0) + 1
        # Duplicate corner IDs are ambiguous; do not count either observation.
        observations = record["charuco_corners"]
        ids = Counter(p["id"] for p in observations)
        for point in observations:
            if point["id"] in self.corners and ids[point["id"]] == 1:
                pixel = np.asarray(point["pixel"], float)
                if pixel.shape != (2,) or not np.isfinite(pixel).all():
                    raise ValueError("Recorded corner pixels must be finite 2D coordinates")
                self.corners[point["id"]].add(pixel)
        if record["pose_available"]:
            self.all_poses.add(record["pose"])
            if record["pose_quality_passed"]:
                self.accepted_poses.add(record["pose"])
        self.processing_times.append(record["processing_ms"])
        host_time = record["host_monotonic_ms"]
        if self.first_time is None:
            self.first_time = host_time
        if self.last_time is not None:
            interval = host_time - self.last_time
            if interval < 0:
                raise ValueError("Recorded host timestamps must be monotonic")
            self.intervals.append(interval)
        self.last_time = host_time
        number = record.get("camera_frame_number")
        if number is not None and self.last_frame_number is not None:
            delta = number - self.last_frame_number
            self.frame_gaps += max(0, delta - 1)
            self.duplicate_frames += delta == 0
            self.frame_number_resets += delta < 0
        self.last_frame_number = number
        timestamp = record.get("camera_timestamp_ms")
        domain = record.get("camera_timestamp_domain")
        if timestamp is not None and self.last_timestamp is not None:
            self.timestamp_resets += domain != self.last_domain or timestamp < self.last_timestamp
        self.last_timestamp, self.last_domain = timestamp, domain

    def summary(self) -> dict:
        span = (self.last_time - self.first_time) / 1000. if self.frames > 1 else 0.
        return {
            "frames_recorded": self.frames,
            "observed_span_seconds": span,
            "observed_processing_fps": (self.frames - 1) / span if span > 0 else None,
            "counts": dict(self.counts),
            "rates": {key: count / self.frames if self.frames else None for key, count in self.counts.items()},
            "longest_dropout_frames": dict(self.longest_dropouts),
            "foreign_marker_frames": self.foreign_frames,
            "duplicate_marker_id_frames": self.duplicate_id_frames,
            "missing_marker_frame_counts": dict(self.missing_markers),
            "quality_reason_frame_counts": dict(self.quality_reasons),
            "camera_frame_number_gaps": self.frame_gaps,
            "duplicate_camera_frames": self.duplicate_frames,
            "camera_frame_number_resets": self.frame_number_resets,
            "camera_timestamp_resets_or_domain_changes": self.timestamp_resets,
            "processing_ms": _distribution(self.processing_times),
            "host_frame_interval_ms": _distribution(self.intervals),
            "corner_jitter": {str(i): moments.summary(self.frames) for i, moments in self.corners.items()},
            "all_poses": self.all_poses.summary(self.reference),
            "quality_passed_poses": self.accepted_poses.summary(self.reference),
            "reference_pose": self.reference,
            "notes": ["Jitter and drift assume a stationary camera and board; no smoothing is applied.",
                      "Rates use every processed frame; board presence must be established independently.",
                      "Sample standard deviation uses N-1; fewer than two observations yield null jitter.",
                      "Coverage is relative to the convex hull of all internal chessboard corners.",
                      "Reprojection error measures image fit; accuracy requires an independent reference pose."],
        }
