"""Validate intrinsics and map supported distortion models to OpenCV."""
import json
from pathlib import Path

import numpy as np


def opencv_calibration(intrinsics: dict, image_pixels=None) -> tuple[np.ndarray, np.ndarray]:
    """Return K and distortion coefficients without silently changing models.

    RealSense inverse/modified Brown-Conrady are not OpenCV Brown-Conrady.
    Zero Brown-Conrady coefficients are equivalent to a pinhole model. Fisheye
    models have different base projections, even with zero coefficients.
    Legacy callers supplying only fx/fy/ppx/ppy explicitly assume no distortion.
    """
    values = np.asarray([intrinsics[k] for k in ("fx", "fy", "ppx", "ppy")], dtype=float)
    if not np.isfinite(values).all() or np.any(values[:2] <= 0):
        raise ValueError("Calibration requires finite fx/fy/ppx/ppy and positive focal lengths")
    if ("width" in intrinsics) != ("height" in intrinsics):
        raise ValueError("Calibration width and height must be provided together")
    if "width" in intrinsics:
        size = [intrinsics["width"], intrinsics["height"]]
        if any(not isinstance(v, (int, float)) or not np.isfinite(v)
               or v <= 0 or int(v) != v for v in size):
            raise ValueError("Calibration width and height must be positive integers")
        if image_pixels is not None and tuple(size) != tuple(image_pixels):
            raise ValueError(f"Calibration size {tuple(size)} does not match image {tuple(image_pixels)}")
    coeffs = np.asarray(intrinsics.get("coeffs", []), dtype=float)
    if coeffs.ndim != 1 or not np.isfinite(coeffs).all():
        raise ValueError("Calibration coeffs must be a finite one-dimensional array")
    model = str(intrinsics.get("model", "none")).removeprefix("distortion.").lower()
    known = {"none", "brown_conrady", "opencv", "inverse_brown_conrady",
             "modified_brown_conrady", "ftheta", "kannala_brandt4"}
    if model not in known:
        raise ValueError(f"Unknown distortion model {model!r}")
    incompatible = (
        f"Distortion model {model!r} is not compatible with OpenCV's standard projection; "
        "supply --calibration with a standard OpenCV calibration for this image/profile, "
        "or rectify the images with the correct model before offline detection")
    if model in {"ftheta", "kannala_brandt4"}:
        raise ValueError(incompatible)
    if model == "none" and np.any(coeffs):
        raise ValueError("Distortion model 'none' cannot have nonzero coefficients")
    if model in {"opencv", "brown_conrady"}:
        allowed = (0, 5) if model == "brown_conrady" else (0, 4, 5, 8, 12, 14)
        if len(coeffs) not in allowed:
            raise ValueError(f"Invalid coefficient count for distortion model {model!r}")
    elif model in {"inverse_brown_conrady", "modified_brown_conrady"}:
        if len(coeffs) not in (0, 5):
            raise ValueError(f"Invalid coefficient count for distortion model {model!r}")
        if np.any(coeffs):
            raise ValueError(incompatible)
    if not np.any(coeffs):
        coeffs = np.zeros(5, dtype=float)
    fx, fy, cx, cy = values
    return np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]]), coeffs


def load_calibration(path: str) -> dict:
    intrinsics = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(intrinsics, dict):
        raise ValueError("Calibration JSON must be an object")
    if "width" not in intrinsics or "height" not in intrinsics:
        raise ValueError("Calibration files must specify width and height for the calibrated profile")
    opencv_calibration(intrinsics)
    return intrinsics
