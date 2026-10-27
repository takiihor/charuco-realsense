#!/usr/bin/env bash
# Stop this user's RealSense camera clients, including a stuck desktop monitor.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$PROJECT_DIR/src/charuco_realsense/processes.py" "$@"
