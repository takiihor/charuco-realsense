"""Marker coordinate labels on the live view and marker selection for pick/place positions."""
import copy

import numpy as np
import pytest

from charuco_realsense.cli import parser

PIL = pytest.importorskip("PIL.Image")


def marker(marker_id, u, v, position=(10., 20., 500.), matched=True, duplicate=False):
    return {"id": marker_id, "matched": matched, "duplicate": duplicate, "pixel_center": [u, v],
            "position_mm": list(position) if position else None}


def changed_pixels(a, b):
    return int(np.any(np.asarray(a) != np.asarray(b), axis=2).sum())


def test_labels_are_drawn_without_overlap_and_crowded_markers_keep_an_id():
    from charuco_realsense.gui import annotate_markers
    base = PIL.new("RGB", (640, 480), (40, 40, 40))
    lone = annotate_markers(base, [marker(1, 320, 240)], .5)
    assert lone.size == base.size and lone.mode == "RGB"
    # The label sits beside the scaled centre (160, 120), not at full-resolution coordinates.
    region = np.asarray(lone)[100:140, 170:260]
    assert changed_pixels(region, np.asarray(base)[100:140, 170:260]) > 200
    crowded = [marker(i, 300 + 6 * i, 240) for i in range(1, 6)]
    image = annotate_markers(base, crowded, 1.)
    assert changed_pixels(image, base) > 0


def test_selected_marker_is_highlighted():
    from charuco_realsense.gui import annotate_markers
    base = PIL.new("RGB", (640, 480), (40, 40, 40))
    markers = [marker(1, 200, 240), marker(2, 400, 240)]
    plain = np.asarray(annotate_markers(base, markers, 1.))
    chosen = np.asarray(annotate_markers(base, markers, 1., selected=2))
    yellow = lambda image: int(np.sum((image[..., 0] > 200) & (image[..., 1] > 170) & (image[..., 2] < 60)))
    assert yellow(chosen) > yellow(plain) + 50


def test_foreign_and_duplicate_markers_are_labelled_without_coordinates():
    from charuco_realsense.gui import annotate_markers
    base = PIL.new("RGB", (640, 480), (40, 40, 40))
    image = annotate_markers(base, [marker(30, 320, 240, None, matched=False), marker(4, 100, 100, None, duplicate=True)], 1.)
    assert changed_pixels(image, base) > 0


@pytest.fixture
def app(tmp_path):
    tk = pytest.importorskip("tkinter")
    pytest.importorskip("PIL.ImageTk")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("No desktop display available")
    root.withdraw()
    from charuco_realsense.gui import MonitorApp
    args = parser().parse_args(["gui", "--board", "front", "--output-dir", str(tmp_path)])
    monitor = MonitorApp(root, args, ["front"], auto_connect=False)
    monitor.connected = True
    yield monitor
    root.destroy()


def test_camera_model_selection_is_passed_to_worker_and_unlocked_after_disconnect(app):
    class Worker:
        def __init__(self, args):
            self.args = args
            self.alive = False

        def start(self):
            self.alive = True

        def is_alive(self):
            return self.alive

        def disconnect(self):
            self.alive = False

    app.worker_factory = Worker
    assert app.camera_model_var.get() == "auto"
    assert tuple(app.camera_model_box["values"]) == ("auto", "D435i", "D415")
    app.camera_model_var.set("D415")
    app.connect_button.invoke()
    assert app.worker.args.camera_model == "D415"
    assert app.camera_model_box.instate(["disabled"])
    assert app.args.camera_model == "auto"
    app._event("camera_lost", {"error": "unplugged", "attempt": 1, "retry_in": 1.})
    assert app.camera_model_box.instate(["disabled"])
    app.connect_button.invoke()
    app._event("disconnected", None)
    assert app.camera_model_box.instate(["readonly", "!disabled"])
    app.camera_model_var.set("D435i")
    app.connect_button.invoke()
    assert app.worker.args.camera_model == "D435i"
    app.worker.disconnect()


def report(markers, passed=True):
    return {"board_recognized": True, "matched_marker_ids": [1, 2], "expected_marker_ids": [1, 2, 3],
            "missing_marker_ids": [3], "foreign_marker_ids": [], "detected_charuco_corner_count": 4,
            "expected_charuco_corner_count": 6, "processing_ms": 3., "pose_quality_passed": passed,
            "pose_quality_reasons": [], "pose": {"position_mm": [-50., -40., 400.], "reprojection_rmse_px": .1},
            "marker_coordinates": markers, "live_jitter": {}}


class Click:
    def __init__(self, x, y):
        self.x, self.y = x, y


def test_clicking_a_marker_shows_and_copies_its_centre(app):
    image = np.zeros((480, 640, 3), np.uint8)
    markers = [marker(1, 100, 100, (-30., -20., 401.)), marker(2, 500, 300, (12.5, 7.25, 399.5))]
    app._show_preview({"report": report(markers), "image": image, "points": []})
    assert app.position_title_var.get() == "BOARD ORIGIN" and app.metrics["x"].get() == "-50.00"
    app._display = (10., 20., .5)  # Image drawn at half size, offset in the canvas.
    app._on_preview_click(Click(10 + 250 + 4, 20 + 150 - 3))
    assert app.selected_marker == 2 and app.position_title_var.get() == "MARKER #2"
    assert [app.metrics[k].get() for k in "xyz"] == ["+12.50", "+7.25", "+399.50"]
    assert app.coordinate_table.selection() == ("2:0",)
    app.copy_coordinates()
    text = app.root.clipboard_get()
    assert text.startswith("Marker 2 centre / camera frame / mm") and "X=+12.50" in text and "Z=+399.50" in text
    # The selection follows later frames; a vanished marker clears the readout instead of going stale.
    app._show_preview({"report": report(markers[:1]), "image": image, "points": []})
    assert app.metrics["x"].get() == "--" and "Not visible" in app.position_note_var.get()
    assert app.copy_button.instate(["disabled"])
    app._on_preview_click(Click(600, 10))  # Empty space returns to the board origin.
    assert app.selected_marker is None and app.metrics["x"].get() == "-50.00"


def test_table_selection_selects_the_marker(app):
    image = np.zeros((480, 640, 3), np.uint8)
    markers = [marker(1, 100, 100, (-30., -20., 401.)), marker(7, 300, 300, None, matched=False)]
    app._show_preview({"report": report(markers, passed=False), "image": image, "points": []})
    app.coordinate_table.selection_set("1:0")
    app.root.update()
    assert app.selected_marker == 1 and app.metrics["x"].get() == "-30.00"
    assert "did not pass quality" in app.position_note_var.get()
    app.coordinate_table.selection_set("7:0")
    app.root.update()
    assert app.selected_marker == 7 and app.metrics["x"].get() == "--"
    assert "Not on the selected board" in app.position_note_var.get()
