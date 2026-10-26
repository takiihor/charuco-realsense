# charuco-realsense

Detect [charuco-generator](../charuco-generator) ChArUco boards with an Intel
RealSense D435i, and estimate the board pose with `cv2.solvePnP`.

The detection core works on any image (PNG/JPG) or on the live color stream.
The camera layer uses `pyrealsense2` and is imported lazily, so offline
workflows and tests run without RealSense hardware or the RealSense runtime.

## Setup

```sh
cd ~/development/charuco-realsense
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e ../charuco-generator --no-deps
.venv/bin/pip install -e .
```

`charuco-generator` is installed with `--no-deps` on purpose: it pins
`opencv-contrib-python-headless`, which conflicts with the GUI `opencv-python`
build this repo needs for `stream`. The generator's runtime imports that this
repo uses (`create_board`, `layout_record`, `reconstruct_board`, `render_png`)
only require `cv2` and `numpy`, which `opencv-python==4.11.0` provides.

## Usage

```sh
# List connected RealSense devices
.venv/bin/python main.py list

# Detect in an image (offline; the default manifest is the sibling checkout)
.venv/bin/python main.py detect --board front \
  --image ../charuco-generator/output/front_charuco.png

# Detect from the camera (best of N frames by corner count), with pose
.venv/bin/python main.py detect --board front --frames 10 --pose

# Live view with marker/corner overlays (q to quit)
.venv/bin/python main.py stream --board front --pose
```

`detect` prints a JSON report and exits `0` when at least one expected marker
is matched, `1` when nothing is found, `2` on configuration/hardware errors.
Use `--report path` to save the JSON and `--save-image path` to save the
analyzed frame. `--serial`, `--width`, `--height`, `--fps` select the device
and stream; `--generator-path` points at a generator checkout when it is not
installed in the environment.

## Stream profiles and USB

The D435i prefers 1280×720@30, but a USB 2.0/2.1 port caps it much lower
(observed: color 1280×720@15, depth 848×480@10). The camera starts the
requested profile, then falls back through
1280×720@15 + 640×480@15 → 640×480@30 → 848×480@10. For full frame rate use a
USB 3.0 (blue) port.

## Report fields

- `expected_marker_ids`, `detected_marker_ids`, `matched_marker_ids`,
  `missing_marker_ids`, `foreign_marker_ids`, `duplicate_marker_ids`
- `detected_charuco_ids`, `detected_charuco_corner_count`,
  `expected_charuco_corner_count`
- `full_expected_visibility` — true only when every marker and every corner is
  detected exactly once
- `pose` (camera mode, `--pose`) — `rotation_matrix`, `rvec`, `tvec_m`,
  `position_mm`; board origin at the outer top-left pattern corner, X right,
  Y down, Z into the board (plane Z=0). `solvePnP` under the supplied
  intrinsics; not a dimensional or metrology certification.
- `source`, `frames_sampled`, `image_pixels`, `rejected_marker_candidate_count`,
  `passed`

## Pose accuracy

`solvePnP` is as good as the corner detection and intrinsics. The RealSense
color intrinsics are factory-calibrated for undistorted pinhole projection;
lens distortion is ignored here, which is fine at moderate distance. Do not
use `position_mm` for dimensional work without a separate calibration and
uncertainty analysis.

## Tests

```sh
.venv/bin/python -m pytest -q
```

Tests render the configured boards from the sibling generator checkout and
verify: full-board detection in the rendered PNG, blank-image rejection,
foreign-board ID separation (rear markers are foreign to the front board), and
pose recovery from a simulated perspective photo.

## Python API

```python
import cv2
from charuco_generator.board import reconstruct_board
from charuco_generator.validation import load_manifest
from charuco_realsense.camera import RealsenseCamera, StreamConfig
from charuco_realsense.detect import detect_charuco
from charuco_realsense.pose import estimate_pose

manifest = load_manifest("../charuco-generator/output/board_manifest.json")
board = reconstruct_board(manifest, "front")
record = next(b for b in manifest["boards"] if b["name"] == "front")

with RealsenseCamera(config=StreamConfig(1280, 720, 15)) as camera:
    frame = camera.read()
    gray = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2GRAY)
    detection = detect_charuco(board, gray, record["geometry"]["border_bits"])
    pose = estimate_pose(board, detection, frame.intrinsics)
```
