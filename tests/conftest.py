import json
from pathlib import Path

import cv2
import pytest
import numpy as np

from charuco_generator.board import canonical_hash, create_board, layout_record
from charuco_generator.config import load_config
from charuco_generator.render import render_png

GENERATOR_ROOT = Path(__file__).resolve().parents[1].parent / "charuco-generator"


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    """Render the configured boards and build a minimal reconstruction manifest.

    Avoids charuco_generator.generate() because its environment() probe requires
    the headless OpenCV distribution, which conflicts with this repo's GUI build.
    """
    config = load_config(GENERATOR_ROOT / "config" / "boards.json")
    directory = tmp_path_factory.mktemp("boards")
    manifest = {
        "schema_version": 1,
        "environment": {"opencv": cv2.__version__},
        "configuration": config.to_dict(),
        "configuration_sha256": canonical_hash(config.to_dict()),
        "boards": [],
    }
    for spec in config.boards:
        board = create_board(config.geometry, spec)
        record = layout_record(config.geometry, spec, board)
        render_png(directory / f"{spec.name}_charuco.png", config.geometry, config.printing, board)
        record["files"] = {"png": f"{spec.name}_charuco.png"}
        manifest["boards"].append(record)
    path = directory / "board_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return directory, json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def known_pose(artifacts):
    """Exact perspective projection with a known pose and nonzero lens distortion."""
    from charuco_generator.board import reconstruct_board
    from charuco_realsense.detect import Detection
    from charuco_realsense.calibration import opencv_calibration

    _, manifest = artifacts
    board = reconstruct_board(manifest, manifest["boards"][0]["name"])
    intrinsics = {"fx": 910., "fy": 900., "ppx": 640., "ppy": 360.,
                  "width": 1280, "height": 720, "model": "brown_conrady",
                  "coeffs": [-.15, .04, .002, -.001, 0.]}
    matrix, coeffs = opencv_calibration(intrinsics)
    rvec, tvec = np.array([.3, -.2, .1]), np.array([.08, -.04, .7])
    points = np.asarray(board.getChessboardCorners(), float)
    pixels, _ = cv2.projectPoints(points, rvec, tvec, matrix, coeffs)
    detection = Detection((1280, 720), (), np.empty((0, 4, 2)),
                          tuple(range(len(points))), pixels, 0)
    return board, detection, intrinsics, rvec, tvec
