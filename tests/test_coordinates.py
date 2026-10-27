from dataclasses import replace

import numpy as np

from charuco_realsense.coordinates import MarkerCoordinates


def test_marker_centres_transform_board_geometry_to_camera_frame(known_pose):
    import cv2
    board, detection, _, rvec, tvec = known_pose
    ids = board.getIds().ravel()
    corners = np.arange(len(ids) * 8).reshape(-1, 4, 2)
    detection = replace(detection, marker_ids=tuple(ids.tolist()), marker_corners=corners)
    rotation = cv2.Rodrigues(rvec)[0]
    rows = MarkerCoordinates(board).measure(detection, {"rotation_matrix": rotation.tolist(), "tvec_m": tvec.tolist()})
    for row, points, pixels in zip(rows, board.getObjPoints(), corners):
        expected = (rotation @ points.mean(axis=0) + tvec) * 1000.
        np.testing.assert_allclose(row["position_mm"], expected, atol=1e-5)
        np.testing.assert_allclose(row["pixel_center"], pixels.mean(axis=0))
        assert row["matched"] and not row["duplicate"]


def test_foreign_duplicate_and_unsolved_markers_have_no_inferred_xyz(known_pose):
    board, detection, _, _, _ = known_pose
    marker_id = int(board.getIds().ravel()[0])
    detection = replace(detection, marker_ids=(marker_id, marker_id, 999), marker_corners=np.zeros((3, 4, 2)))
    coordinates = MarkerCoordinates(board)
    pose = {"rotation_matrix": np.eye(3).tolist(), "tvec_m": [0., 0., 1.]}
    rows = coordinates.measure(detection, pose)
    assert all(row["position_mm"] is None for row in rows)
    assert rows[0]["duplicate"] and rows[1]["duplicate"]
    assert not rows[-1]["matched"]
    assert all(row["position_mm"] is None for row in coordinates.measure(detection, None))
