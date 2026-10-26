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
