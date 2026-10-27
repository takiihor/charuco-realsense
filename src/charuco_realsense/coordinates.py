"""Detected image centres and board-model centres in the camera frame."""
from collections import Counter

import numpy as np


COORDINATE_FRAME = {
    "frame": "camera optical frame",
    "axes": "X right, Y down, Z forward",
    "position_units": "mm",
    "board_origin": "outer top-left pattern corner",
    "marker_position_source": "known board geometry transformed by the raw board pose; not a depth measurement",
    "pixel_centre_source": "arithmetic mean of four detected image corners",
}


class MarkerCoordinates:
    """Cache board geometry once; keep foreign/duplicate detections explicit."""

    def __init__(self, board):
        self.centres = {int(i): np.asarray(points, float).mean(axis=0)
                        for i, points in zip(board.getIds().ravel(), board.getObjPoints())}

    def measure(self, detection, pose):
        counts = Counter(detection.marker_ids)
        rotation = np.asarray(pose["rotation_matrix"]) if pose else None
        translation = np.asarray(pose["tvec_m"]) if pose else None
        result = []
        for marker_id, corners in zip(detection.marker_ids, detection.marker_corners):
            position = None
            matched = marker_id in self.centres
            duplicate = counts[marker_id] > 1
            if pose and matched and not duplicate:
                position = ((rotation @ self.centres[marker_id] + translation) * 1000.).tolist()
            result.append({"id": int(marker_id), "matched": matched, "duplicate": duplicate,
                           "pixel_center": np.asarray(corners).mean(axis=0).tolist(),
                           "position_mm": position})
        return sorted(result, key=lambda p: p["id"])
