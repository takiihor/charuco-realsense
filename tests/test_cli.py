import json
import cv2
import numpy as np
import pytest

from charuco_realsense import cli
from charuco_realsense.camera import CameraError, Frame


@pytest.fixture
def fake_capture(artifacts, monkeypatch):
    directory, manifest = artifacts
    record = manifest["boards"][0]
    image = cv2.imread(str(directory / record["files"]["png"]))
    blank = np.full_like(image, 255)
    height, width = image.shape[:2]
    intrinsics = {"fx": float(width), "fy": float(width), "ppx": width / 2, "ppy": height / 2,
                  "width": width, "height": height, "model": "none", "coeffs": [0.] * 5}

    class Camera:
        def __init__(self):
            self.index = 0
            self.images = [blank, image, blank]
            self.intrinsics = intrinsics
            self.metadata = {"intrinsics": intrinsics, "stream": {"fps": 30}}
            self.closed = False
            self.error = None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def read(self):
            if self.error is not None and self.index == 1:
                raise self.error
            index = self.index
            self.index += 1
            return Frame(self.images[index % 3], intrinsics, index * 100., index + 1, "hardware_clock")

    camera = Camera()
    monkeypatch.setattr(cli, "_open_camera", lambda args: camera)
    common = ["--board", record["name"], "--manifest", str(directory / "board_manifest.json"), "--warmup-seconds", "0"]
    return camera, common, directory


def test_camera_save_image_matches_best_report(fake_capture, tmp_path, capsys):
    camera, common, _ = fake_capture
    path = tmp_path / "best.png"
    assert cli.main(["detect", *common, "--frames", "3", "--pose", "--save-image", str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["frame_index"] == 1
    assert report["camera_frame_number"] == 2
    assert report["pose_available"]
    np.testing.assert_array_equal(cv2.imread(str(path)), camera.images[1])
    assert camera.closed


def test_benchmark_records_failures_images_metadata_and_replays(fake_capture, tmp_path, capsys):
    camera, common, _ = fake_capture
    output = tmp_path / "run"
    assert cli.main(["benchmark", *common, "--frames", "3", "--output", str(output), "--save-frames"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "complete"
    assert summary["frames_recorded"] == 3
    assert summary["counts"]["board_recognized"] == 1
    assert summary["rates"]["pose_available"] == pytest.approx(1 / 3)
    frames = [json.loads(line) for line in (output / "frames.jsonl").read_text().splitlines()]
    assert frames[0]["pose"] is None and frames[2]["pose"] is None
    assert frames[1]["pose"] is not None
    assert frames[1]["charuco_corners"]
    assert len(list((output / "images").glob("*.png"))) == 3
    metadata = json.loads((output / "session.json").read_text())
    assert metadata["effective_intrinsics"] == camera.intrinsics
    assert metadata["pose_options"]["min_corners"] == 6
    assert metadata["detector"]["marker_corner_refinement"] == "none"
    assert cli.main(["summarize", "--record", str(output)]) == 0
    replay = json.loads(capsys.readouterr().out)
    assert replay["counts"] == summary["counts"]
    assert replay["corner_jitter"] == summary["corner_jitter"]
    assert cli.main(["benchmark", *common, "--output", str(output)]) == 2
    assert "already exists" in capsys.readouterr().err


@pytest.mark.parametrize("error,code,status", [(CameraError("unplugged"), 2, "error"),
                                                (KeyboardInterrupt(), 130, "interrupted")])
def test_partial_capture_finalized_on_failure(fake_capture, tmp_path, capsys, error, code, status):
    camera, common, _ = fake_capture
    camera.error = error
    output = tmp_path / "partial"
    assert cli.main(["benchmark", *common, "--output", str(output), "--frames", "3"]) == code
    saved = json.loads((output / "summary.json").read_text())
    assert saved["status"] == status and saved["frames_recorded"] == 1
    assert camera.closed


def test_offline_pose_requires_calibration(fake_capture, tmp_path, capsys):
    camera, common, directory = fake_capture
    image_path = directory / "front_charuco.png"
    assert cli.main(["detect", *common, "--image", str(image_path), "--pose"]) == 2
    assert "requires --calibration" in capsys.readouterr().err
    calibration_path = tmp_path / "intrinsics.json"
    calibration_path.write_text(json.dumps(camera.intrinsics))
    assert cli.main(["detect", *common, "--image", str(image_path), "--pose", "--calibration", str(calibration_path)]) == 0
    assert json.loads(capsys.readouterr().out)["pose_available"]


@pytest.mark.parametrize("arguments", [["--frames", "0"], ["--seconds", "nan"], ["--width", "-1"],
                                        ["--min-coverage", "1.1"], ["--max-rmse-px", "0"]])
def test_invalid_capture_arguments(arguments):
    with pytest.raises(SystemExit) as exc:
        cli.parser().parse_args(["benchmark", "--board", "front", "--output", "unused", *arguments])
    assert exc.value.code == 2


def test_save_image_failure_is_reported(fake_capture, tmp_path, capsys):
    _, common, _ = fake_capture
    path = tmp_path / "absent" / "best.png"
    assert cli.main(["detect", *common, "--frames", "3", "--save-image", str(path)]) == 2
    assert "Cannot save analyzed image" in capsys.readouterr().err


def test_live_stream_can_record_and_quit_without_gui(fake_capture, tmp_path, capsys, monkeypatch):
    camera, common, _ = fake_capture
    monkeypatch.setattr(cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *a: ord("q"))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    output = tmp_path / "live"
    assert cli.main(["stream", *common, "--record", str(output)]) == 1
    saved = json.loads((output / "frames.jsonl").read_text())
    assert saved["pose_requested"]
    assert camera.index == 1 and camera.closed
    assert json.loads(capsys.readouterr().out)["frames_recorded"] == 1


def test_invalid_correspondences_record_as_failed_pose(fake_capture, monkeypatch):
    from dataclasses import replace
    from charuco_generator.board import reconstruct_board
    from charuco_realsense.detect import detect_charuco
    from charuco_realsense.pose import PoseOptions

    camera, _, directory = fake_capture
    manifest = json.loads((directory / "board_manifest.json").read_text())
    board = reconstruct_board(manifest, "front")
    record = manifest["boards"][0]
    detection = detect_charuco(board, cv2.cvtColor(camera.images[1], cv2.COLOR_BGR2GRAY))
    invalid = replace(detection, charuco_ids=(0,) * len(detection.charuco_ids))
    detector = type("Detector", (), {"detect": lambda *a: invalid})()
    _, report = cli._analyze(board, record, detector, camera.images[1], camera.intrinsics, True, PoseOptions())
    assert report["board_recognized"] and not report["pose_available"]
    assert report["pose_quality_reasons"] == ["invalid_corner_correspondences"]
