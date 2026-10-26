import json
from pathlib import Path

import cv2
import pytest

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
