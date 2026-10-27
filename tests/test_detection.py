import cv2
import numpy as np
import pytest

from charuco_generator.board import reconstruct_board
from charuco_realsense.detect import CharucoDetector, Detection, detect_charuco, refine_corners, report_for_board


@pytest.mark.parametrize('calibrated', [False, True])
def test_charuco_pixel_centres_are_unbiased(calibrated):
    """Exact axis-aligned raster: each intersection lies between two pixel centres.

    Test absolute geometry rather than expecting a particular OpenCV bug, so the
    regression also passes when OpenCV fixes its internal coordinate convention.
    """
    board = cv2.aruco.CharucoBoard((5, 7), 0.03, 0.0225,
                                  cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250))
    image = np.pad(board.generateImage((600, 840)), 40, constant_values=255)
    intrinsics = ({'width': 680, 'height': 920, 'fx': 1000., 'fy': 1000.,
                   'ppx': 340., 'ppy': 460., 'model': 'none'} if calibrated else None)
    detection = CharucoDetector(board).detect(image, intrinsics)
    assert len(detection.charuco_ids) == 24
    expected = np.asarray(board.getChessboardCorners())[:, :2] * 4000 + 39.5
    errors = detection.charuco_corners.reshape(-1, 2) - expected[list(detection.charuco_ids)]
    assert np.abs(errors).max() < 0.02
    if calibrated:
        from charuco_realsense.pose import estimate_pose
        pose = estimate_pose(board, detection, intrinsics)
        np.testing.assert_allclose(pose['tvec_m'], [-0.075125, -0.105125, 0.25], atol=1e-6)
    # Applying the image-based correction again does not subtract another 0.5 px.
    repeated = refine_corners(image, detection.charuco_corners)
    np.testing.assert_allclose(repeated, detection.charuco_corners.reshape(-1, 2), atol=1e-3)


def test_refinement_rejects_large_displacement(monkeypatch):
    points = np.array([[20., 20.], [40., 20.], [20., 40.], [40., 40.]], np.float32)
    monkeypatch.setattr(cv2, 'cornerSubPix', lambda image, corners, *args: corners + 2)
    assert refine_corners(np.zeros((60, 60), np.uint8), points) is None


def test_detector_discards_corners_when_refinement_fails(artifacts, monkeypatch):
    directory, manifest = artifacts
    record = manifest['boards'][0]
    board = reconstruct_board(manifest, record['name'])
    image = cv2.imread(str(directory / record['files']['png']), cv2.IMREAD_GRAYSCALE)
    monkeypatch.setattr('charuco_realsense.detect.refine_corners', lambda *args: None)
    detection = CharucoDetector(board).detect(image)
    assert detection.found
    assert detection.charuco_ids == ()
    assert detection.charuco_corners.shape == (0, 1, 2)


def test_detect_generated_png(artifacts):
    directory, manifest = artifacts
    record = manifest["boards"][0]
    board = reconstruct_board(manifest, record["name"])
    image = cv2.imread(str(directory / record["files"]["png"]), cv2.IMREAD_GRAYSCALE)
    detection = detect_charuco(board, image, record["geometry"]["border_bits"])
    assert detection.found
    assert sorted(detection.marker_ids) == sorted(record["marker_ids"])
    assert len(detection.charuco_ids) == record["expected_charuco_corner_count"]
    report = report_for_board(detection, record["marker_ids"], record["expected_charuco_corner_count"])
    assert report["matched_marker_ids"] == sorted(record["marker_ids"])
    assert report["missing_marker_ids"] == []
    assert report["full_expected_visibility"] is True


def test_blank_image_finds_nothing(artifacts):
    directory, manifest = artifacts
    record = manifest["boards"][0]
    board = reconstruct_board(manifest, record["name"])
    detection = detect_charuco(board, cv2.imread(str(directory / record["files"]["png"]),
                                                 cv2.IMREAD_GRAYSCALE) * 0 + 255,
                               record["geometry"]["border_bits"])
    assert not detection.found
    assert detection.charuco_ids == ()
    report = report_for_board(detection, record["marker_ids"], record["expected_charuco_corner_count"])
    assert report["matched_marker_ids"] == []
    assert report["full_expected_visibility"] is False


def test_foreign_ids_reported(artifacts):
    """Rear markers are foreign to the front board (disjoint ID sets)."""
    directory, manifest = artifacts
    front = next(b for b in manifest["boards"] if b["name"] == "front")
    rear = next(b for b in manifest["boards"] if b["name"] == "rear")
    board = reconstruct_board(manifest, "front")
    image = cv2.imread(str(directory / rear["files"]["png"]), cv2.IMREAD_GRAYSCALE)
    detection = detect_charuco(board, image, front["geometry"]["border_bits"])
    report = report_for_board(detection, front["marker_ids"], front["expected_charuco_corner_count"])
    assert report["matched_marker_ids"] == []
    assert report["foreign_marker_ids"] == sorted(rear["marker_ids"])


def test_reusable_detector_passes_markers_once_and_skips_blank(artifacts, monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    from charuco_realsense.detect import CharucoDetector

    directory, manifest = artifacts
    record = manifest["boards"][0]
    board = reconstruct_board(manifest, record["name"])
    image = cv2.imread(str(directory / record["files"]["png"]), cv2.IMREAD_GRAYSCALE)
    real_markers, real_charuco = cv2.aruco.ArucoDetector, cv2.aruco.CharucoDetector
    calls = {"marker_constructors": 0, "charuco_constructors": 0, "marker_passes": 0, "charuco_passes": 0}

    def marker_factory(*args):
        calls["marker_constructors"] += 1
        delegate = real_markers(*args)

        def detect(image):
            calls["marker_passes"] += 1
            return delegate.detectMarkers(image)
        return SimpleNamespace(detectMarkers=detect)

    def charuco_factory(*args):
        calls["charuco_constructors"] += 1
        delegate = real_charuco(*args)

        def detect(image, **kwargs):
            calls["charuco_passes"] += 1
            assert len(kwargs["markerCorners"]) > 0 and len(kwargs["markerIds"]) > 0
            return delegate.detectBoard(image, **kwargs)
        return SimpleNamespace(detectBoard=detect, setCharucoParameters=delegate.setCharucoParameters)

    monkeypatch.setattr(cv2.aruco, "ArucoDetector", marker_factory)
    monkeypatch.setattr(cv2.aruco, "CharucoDetector", charuco_factory)
    detector = CharucoDetector(board)
    assert detector.detect(image).found
    assert not detector.detect(np.full_like(image, 255)).found
    assert detector.detect(image).found
    assert calls == {"marker_constructors": 1, "charuco_constructors": 1,
                     "marker_passes": 3, "charuco_passes": 2}


def test_reusable_detector_resets_calibration(artifacts):
    from charuco_realsense.detect import CharucoDetector

    directory, manifest = artifacts
    record = manifest["boards"][0]
    board = reconstruct_board(manifest, record["name"])
    image = cv2.imread(str(directory / record["files"]["png"]), cv2.IMREAD_GRAYSCALE)
    height, width = image.shape
    detector = CharucoDetector(board)
    intrinsics = {"fx": float(width), "fy": float(width), "ppx": width / 2, "ppy": height / 2,
                  "width": width, "height": height}
    assert detector.detect(image, intrinsics).found
    assert detector._charuco.getCharucoParameters().cameraMatrix is not None
    assert detector.detect(image).found
    assert detector._charuco.getCharucoParameters().cameraMatrix is None
