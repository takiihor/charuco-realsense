"""RealSense D435i detection of charuco-generator ChArUco boards."""

__version__ = "0.1.0"

from .detect import CharucoDetector, Detection, detect_charuco, report_for_board
from .metrics import SessionStatistics
from .pose import PoseOptions, estimate_pose

__all__ = ["CharucoDetector", "Detection", "detect_charuco", "report_for_board",
           "PoseOptions", "estimate_pose", "SessionStatistics"]
