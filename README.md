# charuco-realsense

Detect [charuco-generator](../charuco-generator) ChArUco boards with an Intel
RealSense D435i, estimate pose, and record recognition reliability, corner jitter,
pose jitter, and accuracy against an independent reference pose.

Detection works on PNG/JPG images and the live color stream. `pyrealsense2` is
imported lazily, so offline detection, recording analysis, and tests do not require
camera hardware or the RealSense runtime.

## Setup

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e ../charuco-generator --no-deps
.venv/bin/pip install -e .
```

The generator is installed with `--no-deps` because its headless OpenCV dependency
conflicts with this project's GUI OpenCV build. `./start_camera.sh` can also set
up the environment and run any command below.

The launcher uses `../charuco-generator/output/board_manifest.json` when it exists.
Otherwise it uses the two-stage RealSense **80 mm** board set at
`../charuco-generator/output/two_stage/stage2_charuco_realsense_80mm/board_manifest.json`.
The desktop UI selects `front` when available, or the first board (normally
`tank01` in the two-stage set). It prints the selected manifest and board dimensions
at startup. Pass `--manifest` to use a different generated set; the 32 mm Basler
boards require their own manifest even though they share the 80 mm set's marker IDs.

## Usage

```sh
# Open the desktop UI (also the default for ./start_camera.sh)
./start_camera.sh
# Or explicitly select a board
.venv/bin/python main.py gui --board front

# Select a tank board from the default set
./start_camera.sh --board tank07
# Use the 32 mm boards instead
./start_camera.sh --manifest ../charuco-generator/output/two_stage/stage2_charuco_basler_32mm/board_manifest.json --board tank07

# Desktop UI that starts recording on its own (for unattended stations)
./start_camera.sh gui --board front --auto-record

# Headless 24/7 recorder: reconnects, hourly parts, logs, systemd watchdog
./start_camera.sh service --board front --output-dir output/24x7

# Stop RealSense camera clients if a process is stuck
./kill.sh

.venv/bin/python main.py list

# Offline recognition
.venv/bin/python main.py detect --board front \
  --image ../charuco-generator/output/front_charuco.png

# Keep the best of ten camera frames; save the corresponding image and report
.venv/bin/python main.py detect --board front --frames 10 --pose \
  --save-image output/best.png --report output/detection.json

# Live overlays (q to quit)
.venv/bin/python main.py stream --board front --pose

# Record every frame for 60 seconds, without a GUI
.venv/bin/python main.py benchmark --board front --seconds 60 \
  --output output/run-01

# Record while viewing overlays; pose is enabled automatically
.venv/bin/python main.py stream --board front --record output/run-02

# Optional raw PNG frames, fixed exposure, planar solver and outlier rejection
.venv/bin/python main.py benchmark --board front --frames 500 --seconds 120 \
  --lock-exposure --solver ippe --ransac --refine --save-frames \
  --output output/run-03

# Recompute statistics later, with no hardware
.venv/bin/python main.py summarize --record output/run-01 \
  --report output/run-01/recomputed.json
```

Recording directories must be new; existing experiments are never overwritten.
The recorder creates parent directories. For standalone `detect --report` and
`--save-image`, create the destination parent directory first.

`benchmark` stops at the first reached `--seconds` or `--frames` limit. Its default
is 60 seconds; `--frames` alone therefore still has that time limit. `stream` has
no default limit; it accepts the same optional limits. Ctrl+C finalizes a partial
recording and returns 130. Camera/processing errors also finalize an active
recording with status `error`, then return 2. A hard kill or power failure cannot
run finalization; complete JSONL lines remain available for offline analysis.

`detect` retains its original behavior: exit 0 when any expected marker matches,
1 when none matches, 2 for configuration/hardware/file errors. `benchmark` and a
recorded stream return 0 if at least one recorded frame recognizes the board,
otherwise 1. Recognition success does not imply pose success or accuracy.

## Desktop monitor

Run `./start_camera.sh` to open the UI and connect the camera. Select a board
before connecting, or disconnect to switch boards. The window shows:

- Live marker outlines and IDs, green ChArUco corners, and the board's pose axes
  (X red, Y green, Z blue; 30 mm long).
- Large, signed **X / Y / Z** readouts in **millimetres**, showing the board origin
  in the camera optical frame: **X right, Y down, Z forward**. The board origin
  is its outer top-left pattern corner. **Copy XYZ** includes units and quality.
- Matched-marker/corner counts, image-fit error, processing rate, and plain-language
  quality feedback. Coordinates clear immediately when a pose is unavailable.
- **Coordinates on markers**: each detected marker is labelled in the live view
  with its ID and centre X/Y/Z in mm (camera frame). Labels are placed so they do
  not cover each other; where there is no room a marker keeps a compact `#ID` tag.
  **Click a marker** (or its table row) to select it: it is highlighted, its label
  is always shown, and the large X/Y/Z readout and **Copy XYZ** switch from the
  board origin to that marker's centre. Click empty space to return to the board
  origin. Amber label borders mean the board pose failed its quality checks;
  markers from another board or with duplicate IDs have no coordinates. Untick
  **Coordinates on markers** to hide the labels.
- A **Marker coordinates** tab with each detected ID, board match, image centre
  u/v in pixels, and inferred marker-centre X/Y/Z in camera coordinates.
- Time plots of X/Y/Z displacement from the first pose, position RMS jitter,
  rotation RMS jitter, and corner RMS jitter, each with its latest value. Missing
  observations and camera outages create gaps.
- **Start recording** and **Stop & save** controls, a recording timer/frame count,
  a destination folder picker with free disk space, and an optional raw-PNG checkbox.
- Status pills for the camera (live, reconnecting, error, stalled) and recording
  (`REC hh:mm:ss`, `REC PAUSED` during a camera outage), and an amber banner over
  the preview when frames stop arriving.
- An **Event log** tab (connections, outages, saved parts, warnings; newest first)
  and a status bar with uptime, processed frames, camera frame drops, dropouts,
  processing errors, and the log file path.

Keyboard: **Ctrl+R** starts/stops recording, **F11** toggles full screen,
**Esc** leaves full screen, **Ctrl+Q** quits. The window needs at least 1200 x 800.

Choose the parent destination folder and press **Start recording**. Each click
starts a new `run-YYYYMMDD-HHMMSS-...` directory and resets the live history and
displacement baseline. **Stop & save** finalizes the session while keeping the
preview running. Disconnecting or closing the window also saves an active
recording and releases the camera. If the camera fails or is unplugged, the
current part is saved and the UI reconnects automatically; see
[24/7 operation](#247-operation). **Quit**, **Ctrl+Q** inside the window, and **Ctrl+C** in the launching terminal
also close the camera. SIGINT and SIGTERM save an interrupted recording before
exiting with 130 and 143 respectively. Frame waits check cancellation every
100 ms, including during camera warmup.

UI recordings include the normal `session.json`, `frames.jsonl`, and
`summary.json`, plus **`jitter.csv`** with a row per processed frame: elapsed
seconds, absolute board X/Y/Z in mm, X/Y/Z displacement, rolling position/rotation/corner
jitter, recognition and pose-quality flags, marker/corner counts, hardware timestamp,
and frame number. **`markers.csv`** has one row per detected marker per frame:
elapsed time, frame number, ID, match/duplicate flags, centre u/v, inferred X/Y/Z,
and board-pose quality. Frames with no detected markers have no marker rows;
they remain in `jitter.csv` and `frames.jsonl`. The same live measurements are
stored under `live_jitter` in each JSONL record. Empty CSV cells and JSON nulls
mean insufficient or missing observations, rather than zero jitter.

Marker image centres are the arithmetic mean of their four detected corners.
Marker X/Y/Z are calculated from known board geometry and the raw board pose;
they are **not depth measurements or independent per-marker poses**. They are
in the **camera** optical frame: a robot such as a UR10e needs them transformed
into its base frame with a hand-eye (camera-to-robot) calibration before use as
pick or place targets. Other-board
IDs and duplicate IDs have no inferred X/Y/Z. A solved pose that fails quality
checks remains visible and recorded with its quality flag; it is never silently
presented as accepted. JSONL includes `marker_coordinates`; `session.json` records
the coordinate conventions.

The default rolling window is 2 seconds and the chart shows 30 seconds. Change
them with `gui --rolling-seconds 3 --window-seconds 60`. Rolling position jitter
is the 3D RMS distance from the window's mean position. Rotation jitter is RMS
angular distance from the window's mean rotation. Corner jitter is the square
root of the mean per-ID radial variance for corners observed at least twice in
the window. These use raw solved poses, including quality-rejected poses; the
quality indicator remains separate. Keep the camera and board stationary to
interpret these as jitter. Preview refresh is capped at 10 Hz, independently of
recording every processed frame.

The UI also accepts the usual camera, calibration, solver and reference-pose
options, plus `--output-dir` (default `output`), `--save-frames`, `--auto-record`
and the 24/7 options below. A static
reference pose and any exposure lock remain fixed until disconnecting. UI chart
history is bounded to 5,000 samples; the recording retains all processed frames.

Tkinter and Pillow are required for the desktop UI. The setup above includes
Pillow; package-only installs can use `pip install -e '.[gui]'`. If Tkinter is
missing, install Tk support for the Python interpreter you use (commonly
`python3-tk` on Debian/Ubuntu). The `detect`, `benchmark`, and `summarize` commands
continue to work without Tkinter or a desktop display.

## 24/7 operation

Both `gui` and the headless `service` command are built for unattended use:

- **Automatic reconnection.** A camera error, unplug or 5-second frame timeout
  saves the current recording part with status `error`, then reconnects with
  exponential backoff (1, 2, 4 ... up to `--reconnect-max-seconds`, default 30 s).
  After every `--hardware-reset-after` failed attempts (default 3; 0 disables) an
  idle, enumerable camera receives a RealSense hardware reset. Recording resumes
  in a new part when frames return. `--no-reconnect` restores stop-on-error.
- **Recording parts.** A recording series starts in `run-YYYYMMDD-HHMMSS-ffffff/`.
  Every `--segment-minutes` (default 60; 0 disables) and after each reconnect, the
  next part is written to a sibling `run-...-part0002/`, `-part0003/`, and so on.
  Every part is a complete recording directory (`session.json`, `frames.jsonl`,
  `summary.json`, CSVs) that `summarize` reads on its own. `session.json`
  records `segment: {series, part, reason, segment_minutes}`, with reason `start`,
  `rotation` or `reconnect`. Elapsed time and the displacement baseline continue
  across the parts of a series; `frame_index` and image names restart per part.
  Parts bound per-recording memory (about 300 bytes per frame for statistics,
  ~30 MB per hour at 30 fps) and file sizes.
- **Disk guard.** Recording stops, and the preview continues, when free space
  under the recording folder falls below `--min-free-gb` (default 2 GB). The
  UI shows free space next to the folder (amber below 5x the limit, red below it).
  Budget roughly 4–5 KB of JSONL per frame with a pose (about 12–15 GB/day at
  30 fps, more for boards with many corners) plus the CSVs. Raw PNG frames are
  ~1.5 MB each at 1280 x 720 (about 4 TB/day at 30 fps); use them only for
  short runs. Old recordings are never deleted automatically.
- **Fault isolation.** A frame that fails processing is skipped and counted
  (30 consecutive failures trigger a reconnect). A storage error such as a full
  or removed disk stops the recording but keeps the camera running. A failing
  UI callback is logged and never stops the window's event loop.
- **Logs.** Events go to stderr and a rotating log file (10 MB x 5) at
  `OUTPUT_DIR/logs/charuco-realsense.log`; choose another file with `--log-file`
  or disable it with `--log-file ''`.
- **Hung camera detection.** Recordings are finalized *before* the camera is
  released, because stopping a pipeline on a lost USB device can block. If the
  capture thread makes no progress for `--stall-seconds` (default 30), the UI
  shows **Camera stalled**, and the systemd watchdog stops being fed so the
  service restarts. When quitting, the UI waits up to 20 s for the camera after
  the recording is saved (120 s at most), then exits anyway.

### Running as a service

`service` records without a display until SIGINT/SIGTERM (exit 130/143), or
until a non-camera error (exit 2). `--no-record` only monitors and logs. It
implements the systemd notify protocol (`READY`, `STATUS`, `WATCHDOG`) with
the standard library, so the template in
[`deploy/charuco-realsense.service`](deploy/charuco-realsense.service) runs it
with `Type=notify`, `WatchdogSec=90` and `Restart=always`:

```sh
mkdir -p ~/.config/systemd/user
cp deploy/charuco-realsense.service ~/.config/systemd/user/  # adjust paths/board
systemctl --user daemon-reload
systemctl --user enable --now charuco-realsense
sudo loginctl enable-linger "$USER"   # start at boot without a login
journalctl --user -u charuco-realsense -f
```

To run the desktop UI unattended instead, autostart it in the desktop session,
for example with `~/.config/autostart/charuco-monitor.desktop`:

```ini
[Desktop Entry]
Type=Application
Name=ChArUco monitor
Exec=/home/USER/development/charuco-realsense/start_camera.sh gui --board front --auto-record
```

### Host checklist

- Use a USB 3 port and cable; check that the stream reports the intended profile.
- Disable USB autosuspend, a common cause of RealSense dropouts on long runs:
  add `usbcore.autosuspend=-1` to the kernel command line, or for the current
  boot `echo -1 | sudo tee /sys/module/usbcore/parameters/autosuspend`.
- Disable system sleep/suspend and screen locking that could stop the session.
- Install the librealsense udev rules so the camera is accessible after
  re-enumeration (needed for reconnects and hardware resets).
- Keep the recording folder on a disk sized for the retention you need, and
  archive or delete old `run-*` directories yourself.

## Stop stuck camera processes

```sh
./kill.sh --dry-run  # Inspect matching processes without stopping them
./kill.sh            # SIGINT, wait 5 s, then SIGTERM and finally SIGKILL if needed
./kill.sh --grace 15 # Allow more time to save a large recording
./kill.sh --force    # Immediately SIGKILL matched clients; cannot save a summary
```

This Linux script needs only system Python, not the virtual environment or SDK.
It stops inspectable RealSense clients owned by the current user: this repo's
camera processes, RealSense SDK tools/viewer, and other processes with loaded
`pyrealsense2`/`librealsense2` libraries. It checks executable arguments and loaded
libraries, excludes itself and its caller ancestry, and checks process identity
before signalling. It does not match arbitrary text in terminal/editor commands.
Use the same account that started the camera client.

The script reports each PID and signal. A forced kill skips recording finalization
and exposure-setting restoration; complete JSONL lines can still be analysed with
`summarize`. A process stuck inside an uninterruptible kernel/USB operation can
remain until that operation returns, even after SIGKILL.

## Recording files

Each recording directory contains:

- `session.json`: schema version, UTC start time, board geometry and IDs, manifest
  hash, software versions, camera serial/firmware, requested and negotiated stream
  profiles, native and effective calibration, exposure/gain, detector settings,
  pose thresholds, reference pose, and recording options.
- `frames.jsonl`: one entry per processed frame, including unsuccessful frames.
  Contains hardware frame number, camera timestamp/domain, host monotonic timestamp,
  elapsed time, processing time, detected/missing/foreign/duplicate marker IDs,
  marker coordinates, explicit ChArUco ID/coordinate pairs, raw pose, quality
  reasons, reprojection errors, inliers, coverage, and optional reference errors.
- `summary.json`: completion status and the statistics below.
- `images/00000000.png`, etc., only with `--save-frames`: unannotated BGR captures
  encoded as PNG, referenced by their frame records.

JSONL is flushed after every frame. No temporal smoothing is applied. Logging,
image encoding, capture, and GUI time are excluded from `processing_ms` but affect
observed throughput. Per-corner statistics use online moments; pose and timing
samples remain in memory (compact float arrays, about 300 bytes per frame) for
exact percentiles and rotation means. Long `benchmark`/`stream` recordings
therefore grow memory linearly; the UI and `service` bound it with recording parts.
UI and `service` recordings may also be split into parts; see
[24/7 operation](#247-operation).

## Interpreting the measurements

| Statistic | Meaning |
| --- | --- |
| Recognition, pose-available, quality-pass and full-visibility rates | Counts divided by every processed frame, including failures |
| Longest dropout | Longest consecutive failure run in processed frames |
| Camera frame-number gaps/duplicates/resets | Delivery diagnostics; gaps may include frames skipped while processing |
| Corner jitter | Per-ID mean, sample X/Y standard deviation and radial RMS in pixels, plus observation count and missing rate |
| Position jitter | Sample X/Y/Z standard deviation, 3D RMS/P95 displacement from the mean, axis peak-to-peak spread, and drift in mm |
| Rotation jitter | RMS/P95/max angular deviation from the mean rotation and angular drift in degrees |
| Reprojection error | Distance from an observed corner to its projection through the estimated pose, in pixels |
| Accuracy | Translation bias and error RMS/P95/max in mm, plus angular error RMS/P95/max in degrees, relative to supplied ground truth |

`all_poses` includes every solved raw pose, even if rejected by quality checks.
`quality_passed_poses` shows the accepted subset separately. This makes rejection
rates and the effect of filtering visible. Unsolved poses remain null in the log;
they are never replaced by the preceding pose or zero coordinates.

Sample standard deviations use N-1. Jitter is null for fewer than two observations;
accuracy can be evaluated from one observation. Corners are tracked by ID rather
than array position. Duplicate corner IDs are excluded from corner jitter. The
rotation mean is projected onto SO(3) using SVD; angular differences are computed
from relative rotations, avoiding Euler-angle wraparound. Drift compares the
first and last `max(1, min(30, N // 10))` pose samples, and is expressed at the
configured board origin. Widely dispersed or multimodal orientations can make a
single mean rotation unrepresentative; inspect the raw trajectory in that case.

Recognition rates assume the selected board should be visible throughout the run.
They are not labeled precision/recall or a probability that an ID is correct.
Foreign marker IDs alone are not proof of a false recognition: another board may
actually be visible.

## Pose quality and solver choices

The default iterative solver requires at least four non-collinear corners.
Configurable quality gates default to:

- `--min-corners 6`: minimum inlier corner count.
- `--min-coverage 0.25`: inlier object-point convex-hull area divided by the hull
  area of all internal chessboard corners; this is not the outer printed area.
- `--max-rmse-px 2` and `--max-error-px 5`: maximum RMS and worst inlier residual.
- `--min-inlier-fraction 0.6`: minimum inlier fraction.
- All board chessboard corners must project in front of the camera. Duplicate
  expected marker IDs also fail quality because board association is ambiguous.

These are starting thresholds, not calibrated uncertainty or accuracy guarantees.
Reports distinguish `board_recognized`, `pose_available`, and
`pose_quality_passed`. `passed` remains a recognition-only compatibility field.

`--ransac` requires at least six detected corners. It first identifies inliers
with EPNP RANSAC, then refits with the selected solver. Adjust
`--ransac-threshold-px` (default 3 px) for the camera and image quality. With RANSAC,
quality residuals/coverage use inliers, while `all_corner_reprojection_*` fields
and per-corner error records retain the excluded observations.

`--solver ippe` evaluates both planar pose candidates, prefers positive depth,
and selects the lowest inlier residual. The log retains candidates and their
RMSE gap; similar errors can indicate ambiguous orientations. `--refine` applies
Levenberg-Marquardt refinement to candidates. There is no temporal continuity
prior concealing jumps. Compare solver choices using stationary recordings.

The detector reuses OpenCV detector objects and runs marker detection once per
frame. Camera calibration is passed to ChArUco interpolation. Marker corner
refinement remains disabled; the ChArUco chessboard corners themselves receive
OpenCV's subpixel refinement.

## Calibration

Native RealSense color intrinsics include their distortion model and coefficients
and are captured for the actual negotiated profile. Brown-Conrady coefficients
are supported directly. Brown-Conrady variants with all-zero coefficients are
equivalent to a pinhole model. Nonzero inverse/modified Brown-Conrady and all
fisheye models are rejected instead of silently ignored. Zero fisheye coefficients
still use a different base projection. Use a matching
OpenCV calibration or rectify images with their actual model before offline use.

To override factory intrinsics, pass `--calibration calibration.json`. This is
also required for offline `detect --image ... --pose`. Example format (the values
below are illustrative; replace them with a calibration for your camera):

```json
{
  "width": 1280,
  "height": 720,
  "fx": 904.3,
  "fy": 903.2,
  "ppx": 657.7,
  "ppy": 363.9,
  "model": "opencv",
  "coeffs": [-0.1, 0.02, 0.0, 0.0, 0.0]
}
```

`model: "opencv"` accepts 4/5/8/12/14 standard OpenCV distortion coefficients,
`brown_conrady` accepts five, and `none` requires zero distortion. The file must
include width and height matching the actual image. A USB profile fallback that
changes resolution therefore rejects an incompatible calibration rather than
rescaling it silently. Image size alone cannot verify camera identity or crop;
use a calibration for the selected camera and exact stream profile.

Library callers may still supply only fx/fy/ppx/ppy to assume a pinhole camera.
Providing full calibration is preferred for measurements. The factory calibration
alone does not establish physical measurement accuracy.

## Measuring physical accuracy

1. Measure the printed square spacing and check the board is flat and rigid.
   The sibling generator currently defines a 100 x 80 mm pattern with 20 mm
   squares, but the physical print must match its manifest. A 1% print-scale error
   can cause about 1% distance error with essentially zero reprojection error.
2. Mount both camera and board rigidly. Let exposure settle; warmup defaults to
   two seconds and can be increased with `--warmup-seconds`. Optionally use
   `--lock-exposure` to freeze exposure/gain for the session; original settings
   are restored on exit when the device is still connected.
3. Collect 30–60 seconds per stationary setup. Repeat across working distances,
   tilt angles, lighting, occlusion, and image-center/edge locations. Keep raw
   estimates when comparing quality thresholds or solver choices.
4. Establish an independent reference pose in the **color camera coordinate
   frame**, using the same board origin and axes. The board origin is the outer
   top-left pattern corner, X right, Y down, Z into the board; `tvec_m` is in
   metres. A distance measured from the camera housing is not automatically the
   optical-frame distance. A calibrated translation stage can instead validate
   known relative movements across separately recorded stationary setups.
5. Supply a reference file to the recorder or add it when reanalyzing. A static
   reference is valid only while the camera-to-board pose remains constant.
   Account for the reference instrument's own uncertainty. RealSense depth is a
   separate measurement, not automatic ground truth for this color-based pose.

Example reference format (replace with an independently measured pose):

```json
{
  "tvec_m": [-0.05, -0.04, 0.5],
  "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
}
```

A three-element Rodrigues `rvec` in radians can replace `rotation_matrix`.
Matrices must be finite, orthonormal, and have determinant +1.

```sh
.venv/bin/python main.py benchmark --board front --seconds 60 \
  --reference-pose reference.json --output output/accuracy-01

.venv/bin/python main.py summarize --record output/run-01 \
  --reference-pose reference.json --report output/run-01/accuracy.json
```

The estimated pose from the first frame is useful as a repeatability reference,
but cannot serve as independent ground truth. Low jitter and low reprojection
error can coexist with substantial position bias.

## Stream profiles and USB

Capture binds the selected `--serial` on every startup/fallback attempt. Color-only
capture is the default because depth is unused by the estimator. Use `--depth` to
also enable the depth stream; no depth values are currently fused into pose.

The camera first tries the requested profile (default 1280x720@30), then
1280x720@15, 640x480@30, and 848x480@10. When depth is enabled, fallback depth
profiles are respectively 640x480@15, 640x480@30, and 848x480@10. The actual profile
is recorded. The tested USB 2.1 connection negotiates color 1280x720@15; USB 3.0
provides more bandwidth. Frame-number gaps should be considered when comparing
runs with different capture or image-saving settings.

## Tests

```sh
.venv/bin/python -m pytest -q
```

Tests cover rendered-board recognition and foreign IDs; known-pose projection
with lens distortion; iterative/IPPE/refined solvers; outlier rejection, coverage
and residual quality gates; print-scale bias; corner-ID association and missing
observations; rotation wraparound; analytic jitter/accuracy statistics; camera
serial/fallback/error handling; exposure restoration; best-image selection;
recording/reanalysis; and partial-recording finalization. Synthetic checks verify
implementation behavior, not a hardware accuracy specification.

## Python API

```python
import cv2
from charuco_generator.board import reconstruct_board
from charuco_generator.validation import load_manifest
from charuco_realsense import CharucoDetector, PoseOptions, estimate_pose
from charuco_realsense.camera import RealsenseCamera

manifest = load_manifest("../charuco-generator/output/board_manifest.json")
board = reconstruct_board(manifest, "front")
record = next(b for b in manifest["boards"] if b["name"] == "front")
detector = CharucoDetector(board, record["geometry"]["border_bits"])

with RealsenseCamera() as camera:
    frame = camera.read()
    gray = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2GRAY)
    detection = detector.detect(gray, frame.intrinsics)
    pose = estimate_pose(board, detection, frame.intrinsics, PoseOptions())
    if pose is not None:
        print(pose["position_mm"], pose["quality_passed"], pose["reprojection_rmse_px"])
```

The single-image `detect_charuco(board, gray, border_bits, intrinsics=None)` helper
remains available. For sequences, reuse `CharucoDetector` as above.

The detector re-refines ChArUco chessboard intersections on the original image to correct the
approximately +0.5 px offset in OpenCV 4.11's `detectBoard` output
([OpenCV issue #25539](https://github.com/opencv/opencv/issues/25539)). Marker corners are unchanged.
The window scales with corner spacing and frames with a refinement displacement over one pixel
lose their ChArUco corners, so they cannot produce a pose from that suspect refinement. This
applies to still-image detection, streaming, the monitor and headless service. Session metadata
records `charuco_corner_refinement: true`. Use `CharucoDetector(board, refine=False)` only for
diagnostics or when the caller performs its own image-based refinement.

References: [OpenCV ChArUco detection](https://docs.opencv.org/4.11.0/df/d4a/tutorial_charuco_detection.html),
[OpenCV pose solvers](https://docs.opencv.org/4.11.0/d5/d1f/calib3d_solvePnP.html),
[RealSense projection and distortion models](https://dev.realsenseai.com/docs/projection-in-realsense-sdk-2-0/).
