#!/usr/bin/env bash
# Set up this checkout and run the RealSense detection CLI. No activation or global pip needed.
#
#   ./start_camera.sh                 # live stream view of the front board
#   ./start_camera.sh --board rear    # live stream view of another board
#   ./start_camera.sh detect --board front --frames 10 --pose
#   ./start_camera.sh list
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$PROJECT_DIR"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
GENERATOR_DIR="$PROJECT_DIR/../charuco-generator"

python_is_supported() {
    "$1" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))' >/dev/null 2>&1
}

if [[ ! -x "$VENV_PYTHON" ]]; then
    SELECTED_PYTHON=""
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        if command -v "$PYTHON_BIN" >/dev/null 2>&1 && python_is_supported "$PYTHON_BIN"; then
            SELECTED_PYTHON="$PYTHON_BIN"
        else
            printf 'Error: PYTHON_BIN must point to a working Python 3.11+ interpreter.\n' >&2
            exit 1
        fi
    else
        # Prefer the tested Python 3.12 when available, then other supported versions.
        for candidate in python3.12 python3.11 python3.13 python3.14 python3; do
            if command -v "$candidate" >/dev/null 2>&1 && python_is_supported "$candidate"; then
                SELECTED_PYTHON="$candidate"
                break
            fi
        done
    fi
    if [[ -z "$SELECTED_PYTHON" ]]; then
        printf 'Error: install Python 3.11+ with venv support, then run this script again.\n' >&2
        printf 'You can choose an interpreter with PYTHON_BIN=/path/to/python ./start_camera.sh\n' >&2
        exit 1
    fi
    printf 'Creating isolated environment with %s...\n' "$SELECTED_PYTHON"
    if ! "$SELECTED_PYTHON" -m venv "$PROJECT_DIR/.venv"; then
        printf 'Error: could not create .venv. Install venv support for your Python (python3-venv on Debian/Ubuntu).\n' >&2
        exit 1
    fi
fi

if ! python_is_supported "$VENV_PYTHON"; then
    printf 'Error: the existing .venv needs Python 3.11+. Move it aside and rerun with PYTHON_BIN=/path/to/python.\n' >&2
    exit 1
fi

if ! "$VENV_PYTHON" -m pip --version >/dev/null 2>&1; then
    "$VENV_PYTHON" -m ensurepip --upgrade
fi

# Avoid invoking pip or contacting the network when every pinned package is installed.
if ! "$VENV_PYTHON" - "$PROJECT_DIR/requirements.txt" <<'PY'
import importlib.metadata
from pathlib import Path
import sys

try:
    from packaging.requirements import Requirement
    for line in Path(sys.argv[1]).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        if not requirement.specifier.contains(importlib.metadata.version(requirement.name)):
            raise ValueError(requirement.name)
except (ImportError, importlib.metadata.PackageNotFoundError, ValueError):
    raise SystemExit(1)
PY
then
    printf 'Installing pinned dependencies...\n'
    export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PROJECT_DIR/.venv/pip-cache}"
    if ! "$VENV_PYTHON" -m pip install --disable-pip-version-check --quiet -r "$PROJECT_DIR/requirements.txt"; then
        printf 'Error: dependency installation failed. Check the pip error above and your internet connection, then rerun.\n' >&2
        exit 1
    fi
fi

# Install this package editable, and the sibling generator without its own
# dependencies (its headless OpenCV pin conflicts with the GUI build needed
# for the stream command).
if ! "$VENV_PYTHON" -c 'import charuco_realsense' >/dev/null 2>&1; then
    printf 'Installing charuco-realsense (editable)...\n'
    "$VENV_PYTHON" -m pip install --disable-pip-version-check --quiet -e "$PROJECT_DIR"
fi

if [[ ! -d "$GENERATOR_DIR" ]]; then
    printf 'Note: %s not found; pass --generator-path to the generator checkout.\n' "$GENERATOR_DIR"
elif ! "$VENV_PYTHON" -c 'import charuco_generator' >/dev/null 2>&1; then
    printf 'Installing charuco-generator (editable, --no-deps)...\n'
    "$VENV_PYTHON" -m pip install --disable-pip-version-check --quiet -e "$GENERATOR_DIR" --no-deps
fi

if [[ $# -eq 0 ]]; then
    set -- stream --board front
elif [[ "$1" == -* && "$1" != "--help" && "$1" != "-h" ]]; then
    # Convenience: ./start_camera.sh --board rear
    set -- stream "$@"
fi

exec "$VENV_PYTHON" "$PROJECT_DIR/main.py" "$@"
