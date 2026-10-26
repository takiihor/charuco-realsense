"""Board pose from detected ChArUco corners and camera intrinsics (solvePnP)."""
import numpy as np

import cv2

from .detect import Detection


def estimate_pose(board, detection: Detection, intrinsics: dict) -> dict | None:
    """Estimate the board pose in camera coordinates.

    ``intrinsics`` needs ``fx``, ``fy``, ``ppx``, ``ppy`` (RealSense color
    intrinsics). Returns None when fewer than four non-collinear corners are
    available or solvePnP fails.

    Coordinate frame (from the generator manifest): origin at the outer
    top-left pattern corner, X right along columns, Y down along rows, Z into
    the board (board plane at Z=0).
    """
    ids = np.asarray(detection.charuco_ids, dtype=np.int32)
    if len(ids) < 4 or board.checkCharucoCornersCollinear(ids):
        return None
    object_points = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    selected = object_points[list(ids)]
    image_points = np.asarray(detection.charuco_corners, dtype=np.float64)
    camera_matrix = np.array([
        [intrinsics["fx"], 0.0, intrinsics["ppx"]],
        [0.0, intrinsics["fy"], intrinsics["ppy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        selected, image_points, camera_matrix, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rvec = np.asarray(rvec, dtype=np.float64).ravel()
    tvec_m = np.asarray(tvec, dtype=np.float64).ravel()
    rotation, _ = cv2.Rodrigues(rvec)
    return {
        "ok": True,
        "corner_count": int(len(ids)),
        "rotation_matrix": rotation.tolist(),
        "rvec": rvec.tolist(),
        "tvec_m": tvec_m.tolist(),
        "position_mm": (tvec_m * 1000.0).tolist(),
        "board_origin": "outer top-left pattern corner; X right, Y down, Z into board (plane Z=0)",
        "units": "rotation/rvec dimensionless; tvec_m metres; position_mm millimetres",
        "note": "solvePnP result under the supplied intrinsics; not a dimensional or metrology certification",
    }
