import json

import cv2
import numpy as np
import pytest

from charuco_realsense.metrics import (SessionStatistics, rotation_angle_deg,
                                       validate_reference_pose)
from charuco_realsense.recording import SessionRecorder, summarize_recording


def pose(position=(0., 0., .5), angle=0., accepted=True):
    rotation, _ = cv2.Rodrigues(np.array([0., 0., np.radians(angle)]))
    return {"tvec_m": list(position), "rotation_matrix": rotation.tolist(),
            "reprojection_rmse_px": .2, "all_corner_reprojection_rmse_px": .3,
            "board_coverage_fraction": 1., "inlier_count": 12,
            "quality_passed": accepted}


def record(index=0, estimated_pose=None, corners=None, number=None):
    return {"frame_index": index, "host_monotonic_ms": index * 100.,
            "camera_frame_number": index + 1 if number is None else number,
            "camera_timestamp_ms": index * 100., "camera_timestamp_domain": "hardware_clock",
            "board_recognized": estimated_pose is not None,
            "pose_available": estimated_pose is not None,
            "pose_quality_passed": estimated_pose["quality_passed"] if estimated_pose else False,
            "pose_quality_reasons": [] if estimated_pose else ["pose_unsolved"],
            "full_expected_visibility": estimated_pose is not None,
            "foreign_marker_ids": [], "duplicate_marker_ids": [],
            "missing_marker_ids": [] if estimated_pose else [0, 1],
            "charuco_corners": corners or [], "pose": estimated_pose, "processing_ms": 5.}


def test_raw_and_accepted_statistics_and_id_alignment():
    stats = SessionStatistics(2, {"tvec_m": [0, 0, .5], "rotation_matrix": np.eye(3).tolist()})
    frames = [record(0, pose(), [{"id": 0, "pixel": [10, 20]}, {"id": 1, "pixel": [30, 40]}]),
              record(1),
              record(2, pose((.002, .004, .506), accepted=False),
                     [{"id": 1, "pixel": [34, 42]}, {"id": 0, "pixel": [12, 24]}]),
              record(3), record(4)]
    for frame in frames:
        stats.add(frame)
    summary = stats.summary()
    assert summary["rates"]["pose_available"] == .4
    # The rejected estimate contributes to raw jitter and accuracy.
    assert summary["all_poses"]["samples"] == 2
    assert summary["quality_passed_poses"]["samples"] == 1
    assert summary["longest_dropout_frames"]["pose_available"] == 2
    np.testing.assert_allclose(summary["all_poses"]["position_jitter"]["std_xyz_mm"], np.sqrt([2, 8, 18]))
    assert summary["all_poses"]["position_jitter"]["rms_displacement_mm"] == pytest.approx(np.sqrt(14))
    np.testing.assert_allclose(summary["all_poses"]["accuracy"]["translation_bias_xyz_mm"], [1, 2, 3])
    assert summary["all_poses"]["accuracy"]["translation_error_norm_mm"]["rms"] == pytest.approx(np.sqrt(28))
    assert summary["quality_passed_poses"]["position_jitter"]["std_xyz_mm"] is None
    assert summary["corner_jitter"]["0"]["samples"] == 2
    assert summary["corner_jitter"]["0"]["missing_rate"] == .6
    np.testing.assert_allclose(summary["corner_jitter"]["0"]["std_xy_px"], np.sqrt([2, 8]))
    np.testing.assert_allclose(summary["corner_jitter"]["1"]["mean_pixel"], [32, 41])
    json.dumps(summary, allow_nan=False)


def test_rotation_wraparound_is_small_jitter():
    stats = SessionStatistics(0)
    stats.add(record(0, pose(angle=179)))
    stats.add(record(1, pose(angle=-179)))
    result = stats.summary()["all_poses"]["rotation_jitter"]
    assert result["angular_deviation_deg"]["rms"] == pytest.approx(1.)
    assert result["drift_deg"] == pytest.approx(2.)
    assert rotation_angle_deg(np.eye(3), np.eye(3)) == 0.


def test_frame_gaps_duplicates_and_empty_observations():
    stats = SessionStatistics(1)
    for i, number in enumerate([1, 2, 4, 4, 1]):
        stats.add(record(i, number=number))
    result = stats.summary()
    assert result["camera_frame_number_gaps"] == 1
    assert result["duplicate_camera_frames"] == 1
    assert result["camera_frame_number_resets"] == 1
    assert result["longest_dropout_frames"]["board_recognized"] == 5
    assert result["corner_jitter"]["0"]["missing_rate"] == 1.
    assert result["corner_jitter"]["0"]["std_xy_px"] is None
    assert result["all_poses"]["position_jitter"] is None
    assert SessionStatistics(1).summary()["rates"]["board_recognized"] is None
    json.dumps(result, allow_nan=False)


def test_duplicate_corner_ids_are_excluded():
    stats = SessionStatistics(1)
    stats.add(record(corners=[{"id": 0, "pixel": [1, 2]}, {"id": 0, "pixel": [5, 6]}]))
    assert stats.summary()["corner_jitter"]["0"]["samples"] == 0


@pytest.mark.parametrize("rotation", [np.zeros((3, 3)), np.diag([1., 1., -1.]), np.ones((2, 2))])
def test_reference_must_be_proper_rotation(rotation):
    with pytest.raises(ValueError, match="proper 3x3"):
        validate_reference_pose({"tvec_m": [0, 0, .5], "rotation_matrix": rotation.tolist()})


@pytest.mark.parametrize("error,status", [(KeyboardInterrupt(), "interrupted"), (RuntimeError("unplugged"), "error")])
def test_recording_finalizes_on_error_and_reanalyzes(tmp_path, error, status):
    directory = tmp_path / "run"
    metadata = {"board": {"expected_charuco_corner_count": 1}}
    with pytest.raises(type(error)):
        with SessionRecorder(str(directory), metadata) as recorder:
            recorder.add(record(0, pose()))
            recorder.add(record(1))
            raise error
    saved = json.loads((directory / "summary.json").read_text())
    assert saved["status"] == status
    assert saved["frames_recorded"] == 2
    replay = summarize_recording(str(directory), {"tvec_m": [0, 0, .5], "rvec": [0, 0, 0]})
    assert replay["frames_recorded"] == saved["frames_recorded"]
    assert replay["all_poses"]["accuracy"]["translation_error_norm_mm"]["rms"] == 0.
    with pytest.raises(FileExistsError):
        SessionRecorder(str(directory), metadata)
