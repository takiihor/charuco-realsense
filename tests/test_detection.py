import cv2

from charuco_generator.board import reconstruct_board
from charuco_realsense.detect import Detection, detect_charuco, report_for_board


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
