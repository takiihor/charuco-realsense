# charuco-realsense

Live ChArUco detection, camera-frame coordinates, and repeatability measurements
with an Intel RealSense D435i and boards from
[charuco-generator](https://github.com/takiihor/charuco-generator).

## Start

Keep the generator checkout beside this repository and generate your boards first.
You need Python 3.11+ with venv support, plus Tkinter for the desktop UI.

```sh
./start_camera.sh                 # Open the monitor and connect the camera
./start_camera.sh --board tank07  # Select another board in the default set
```

The launcher creates `.venv` and installs dependencies automatically. It prefers
Python 3.12 when creating the environment; no manual activation is needed.
The selected manifest and board dimensions are logged at startup.

Default manifests are checked in this order:

1. `../charuco-generator/output/board_manifest.json` (legacy Front/Rear set).
2. `../charuco-generator/output/two_stage/stage2_charuco_realsense_80mm/board_manifest.json`.

The UI selects `front` if present, otherwise the first board (`tank01` in the
two-stage set). To choose a different set, pass its manifest explicitly:

```sh
./start_camera.sh --board tank01 \
  --manifest ../charuco-generator/output/two_stage/stage2_charuco_realsense_80mm/board_manifest.json
```

Use the manifest matching the physical board. The 32 mm Basler and 80 mm RealSense
sets share marker IDs but have different geometry. For the 80 mm set, the
**checkerboard pattern measures 80 × 80 mm, excluding the white margin**, with
**20 mm squares** and **15 mm markers**.

## Use the desktop monitor

- **X / Y / Z** are millimetres in the camera frame: X right, Y down, Z forward.
  The default point is the checkerboard's outer top-left corner.
- Click a marker or its table row to show its centre coordinates; click empty
  preview space to return to the board origin. **Copy XYZ** includes units and quality.
- Check marker/corner counts, image-fit error, and pose quality. Marker coordinates
  come from the board pose; they are not independent depth measurements.
- Keep both camera and board fixed to interpret the plots as jitter.
- Choose a recording folder, click **Start recording**, then **Stop & save**.
  Saving keeps the live preview running. Disconnect before switching boards.

**Ctrl+R** toggles recording, **F11** toggles full screen, **Esc** leaves full
screen, and **Ctrl+Q** quits. Closing the monitor saves an active recording.

## Record and analyze

Record a stationary sample without the desktop UI, then recompute its statistics:

```sh
./start_camera.sh benchmark --board tank01 --seconds 60 --output output/run-01
./start_camera.sh summarize --record output/run-01
```

Use a new recording directory each time. Benchmark recordings include every
processed frame, including detection failures; no temporal smoothing is applied.
For `detect`, `stream`, `benchmark`, and `service`, `--board` is required.

| File | Contents |
| --- | --- |
| `session.json` | Board geometry, camera profile, calibration, and settings |
| `frames.jsonl` | Per-frame detections, raw poses, quality checks, and timestamps |
| `summary.json` | Recognition rates, jitter, drift, and optional reference-pose errors |
| `jitter.csv`, `markers.csv` | Additional per-frame exports from the monitor and service |
| `images/` | Raw PNGs, only when `--save-frames` is enabled |

Other useful commands:

```sh
./start_camera.sh list
./start_camera.sh stream --board tank01 --pose
./start_camera.sh detect --board tank01 \
  --image ../charuco-generator/output/two_stage/stage2_charuco_realsense_80mm/tank01_charuco.png
./start_camera.sh gui --auto-record
```

Use `./start_camera.sh COMMAND --help` for options. Offline image pose estimation
requires `--calibration`; see the [reference guide](docs/REFERENCE.md#calibration).

## Measured 80 mm board result

On **2026-10-28**, a fixed D435i observed a fixed `tank01` board at approximately
269 mm Z, using factory intrinsics, automatic exposure, and color 1280 × 720 at
15 fps over USB 2.1. The sample contained **1,048 frames over 69.9 seconds**.

| Measurement | Result |
| --- | ---: |
| Markers / corners visible in every frame | 8/8 markers, 9/9 corners |
| Recognition and pose-quality pass rates | 100% |
| Mean / maximum reprojection RMS | 0.096 / 0.146 px |
| Board-origin X / Y / Z standard deviation | 0.034 / 0.012 / 0.154 mm |
| Board-origin 3D RMS variation | 0.158 mm |
| Board-origin 95th-percentile displacement from the mean | 0.303 mm |
| Board-origin Z peak-to-peak range | 1.083 mm |
| Orientation RMS variation | 0.200° |
| Actual frame rate / camera frame gaps | 14.99 fps / 0 |

The modelled board centre had 0.068 mm 3D RMS variation; it is derived from the
same pose, at board-local coordinates `[40, 40, 0]` mm. Coordinate stability depends
on the chosen point as well as the pose's orientation stability.

These are **short-term repeatability results from one setup**. Absolute accuracy
was not measured against an independent reference. Verify the printed scale and
camera calibration before relying on physical distances: a 1% board-scale error
can produce about 2.7 mm depth bias at this range despite a low image-fit error.
See the [saved measurement summary](docs/measurements/2026-10-28-80mm.json) and
[accuracy measurement procedure](docs/REFERENCE.md#measuring-physical-accuracy).

## Continuous operation

```sh
./start_camera.sh service --board tank01 --output-dir output/24x7
```

The monitor and service reconnect after camera failures, split recordings into
hourly parts, and stop recording below 2 GB free space by default. Old recordings
are never deleted automatically. Events are logged to
`OUTPUT_DIR/logs/charuco-realsense.log`.

For systemd setup, watchdogs, storage planning, calibration, solver options, and
the Python API, see the [reference guide](docs/REFERENCE.md).

## Troubleshooting

| Problem | Action |
| --- | --- |
| Missing manifest or unknown board | Generate the board set or pass its `--manifest` and a board name from that set. |
| Desktop UI cannot open | Install Tk support for the selected Python and use a desktop display; use `service` for headless recording. |
| Markers detected but pose unavailable or rejected | Show more of the board and check the quality message; the default quality gate needs at least 6 corners. |
| Lower frame rate than requested | Check the negotiated stream profile and USB connection. The tested USB 2.1 connection uses 1280 × 720 at 15 fps. |
| Camera busy or stuck | Inspect with `./kill.sh --dry-run`, then use `./kill.sh` to stop matching clients. This stops their active capture. |

## Development and CI

```sh
.venv/bin/python -m pytest -q
```

GitHub Actions runs on a self-hosted Linux x64 runner with the `charuco-realsense`
label. [CI](.github/workflows/ci.yml) creates an isolated environment, checks package
dependencies, installs the sibling generator, and runs the tests. Desktop tests
skip when Tkinter or a display is unavailable. Tests and offline analysis do not
require camera hardware.
