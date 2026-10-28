"""Startup must use the generated board set without assuming a front board exists."""
import json

import pytest

from charuco_realsense import cli


@pytest.mark.parametrize("legacy_exists", [False, True])
def test_gui_finds_default_manifest(artifacts, tmp_path, monkeypatch, legacy_exists):
    from charuco_realsense import gui

    _, manifest = artifacts
    generator = tmp_path / "charuco-generator"
    output = generator / "output"
    realsense = output / "two_stage" / "stage2_charuco_realsense_80mm" / "board_manifest.json"
    realsense.parent.mkdir(parents=True)
    realsense.write_text(json.dumps(manifest))
    legacy = output / "board_manifest.json"
    if legacy_exists:
        legacy.write_text(json.dumps(manifest))
    expected = legacy if legacy_exists else realsense
    selections = []

    def launch(args):
        assert args.manifest == str(expected)
        _, record, _ = cli._load_board(args.manifest, args.board, args.generator_path)
        selections.append(record["name"])
        return 0

    monkeypatch.setattr(gui, "run_gui", launch)
    assert cli.main(["gui", "--generator-path", str(generator)]) == 0
    assert selections == ["front"]


@pytest.mark.parametrize("names,selected", [(["rear", "front"], "front"), (["rear"], "rear")])
def test_gui_selects_an_existing_board(artifacts, tmp_path, names, selected):
    _, manifest = artifacts
    manifest = {**manifest, "boards": [next(b for b in manifest["boards"] if b["name"] == name)
                                     for name in names]}
    path = tmp_path / "board_manifest.json"
    path.write_text(json.dumps(manifest))
    _, record, _ = cli._load_board(str(path), None, None)
    assert record["name"] == selected


def test_explicit_board_is_preserved_and_unknown_board_lists_choices(artifacts):
    directory, _ = artifacts
    path = str(directory / "board_manifest.json")
    _, record, _ = cli._load_board(path, "rear", None)
    assert record["name"] == "rear"
    with pytest.raises(ValueError, match="Unknown board 'tank01'.*Available boards: front, rear"):
        cli._load_board(path, "tank01", None)


def test_explicit_missing_manifest_does_not_fall_back(artifacts, tmp_path, capsys):
    _, manifest = artifacts
    output = tmp_path / "output"
    output.mkdir()
    (output / "board_manifest.json").write_text(json.dumps(manifest))
    missing = tmp_path / "missing.json"
    assert cli.main(["gui", "--manifest", str(missing), "--generator-path", str(tmp_path)]) == 2
    assert f"Cannot read manifest {missing}" in capsys.readouterr().err


def test_missing_defaults_do_not_select_basler_geometry(tmp_path, capsys):
    output = tmp_path / "output" / "two_stage" / "stage2_charuco_basler_32mm"
    output.mkdir(parents=True)
    (output / "board_manifest.json").write_text("{}")
    assert cli.main(["gui", "--generator-path", str(tmp_path)]) == 2
    message = capsys.readouterr().err
    assert "No default ChArUco manifest found" in message
    assert "--manifest PATH --board NAME" in message


@pytest.mark.parametrize("command,extra", [("detect", []), ("stream", []), ("benchmark", ["--output", "unused"]),
                                          ("service", [])])
def test_unattended_commands_still_require_explicit_board(command, extra):
    with pytest.raises(SystemExit) as exc:
        cli.parser().parse_args([command, *extra])
    assert exc.value.code == 2
