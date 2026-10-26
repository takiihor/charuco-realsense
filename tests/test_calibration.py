import json

import numpy as np
import pytest

from charuco_realsense.calibration import load_calibration, opencv_calibration


def calibration(**changes):
    return {"fx": 900., "fy": 901., "ppx": 640., "ppy": 360.,
            "width": 1280, "height": 720, "model": "brown_conrady",
            "coeffs": [.1, -.02, .001, .002, 0.], **changes}


def test_brown_conrady_and_profile_validation():
    matrix, coeffs = opencv_calibration(calibration(), (1280, 720))
    np.testing.assert_array_equal(matrix, [[900, 0, 640], [0, 901, 360], [0, 0, 1]])
    np.testing.assert_array_equal(coeffs, [.1, -.02, .001, .002, 0])
    with pytest.raises(ValueError, match="does not match"):
        opencv_calibration(calibration(), (640, 480))


@pytest.mark.parametrize("model", ["inverse_brown_conrady", "modified_brown_conrady"])
def test_unsupported_nonzero_distortion_never_silently_ignored(model):
    with pytest.raises(ValueError, match="not compatible"):
        opencv_calibration(calibration(model=model))
    _, coeffs = opencv_calibration(calibration(model=model, coeffs=[0.] * 5))
    assert not coeffs.any()


@pytest.mark.parametrize("model", ["ftheta", "kannala_brandt4"])
def test_fisheye_base_projection_is_not_pinhole_even_with_zero_coefficients(model):
    with pytest.raises(ValueError, match="not compatible"):
        opencv_calibration(calibration(model=model, coeffs=[0.] * 4))


@pytest.mark.parametrize("changes", [
    {"fx": 0}, {"fy": float("nan")}, {"ppy": float("inf")},
    {"coeffs": [float("nan")] * 5}, {"model": "none"},
    {"model": "unknown"}, {"coeffs": [1, 2, 3]}, {"width": 0},
])
def test_invalid_calibration(changes):
    with pytest.raises(ValueError):
        opencv_calibration(calibration(**changes))


def test_calibration_file_requires_size(tmp_path):
    path = tmp_path / "calibration.json"
    values = calibration()
    path.write_text(json.dumps(values))
    assert load_calibration(str(path)) == values
    del values["width"]
    del values["height"]
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="width and height"):
        load_calibration(str(path))
