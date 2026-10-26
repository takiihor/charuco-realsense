"""Intel RealSense (D435i) capture via pyrealsense2.

pyrealsense2 is imported lazily so image-only workflows (detect --image, tests)
run without the RealSense runtime or hardware.
"""
from dataclasses import dataclass

import numpy as np


class CameraError(RuntimeError):
    pass


@dataclass(frozen=True)
class StreamConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass(frozen=True)
class Frame:
    bgr: np.ndarray
    intrinsics: dict  # fx, fy, ppx, ppy, width, height
    timestamp_ms: float


def _realsense():
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise CameraError(
            "pyrealsense2 is not installed; run: pip install pyrealsense2") from exc
    return rs


def list_devices() -> list[dict]:
    rs = _realsense()
    devices = []
    for device in rs.context().query_devices():
        devices.append({
            "name": device.get_info(rs.camera_info.name),
            "serial_number": device.get_info(rs.camera_info.serial_number),
            "firmware": device.get_info(rs.camera_info.firmware_version),
            "physical_port": device.get_info(rs.camera_info.physical_port),
            "usb_type": device.get_info(rs.camera_info.usb_type_descriptor),
        })
    return devices


# (color_w, color_h, color_fps, depth_w, depth_h, depth_fps); USB 2.0 ports cap
# the D435i well below 1280x720@30, so fall back until one profile resolves.
_FALLBACKS = (
    (1280, 720, 15, 640, 480, 15),
    (640, 480, 30, 640, 480, 30),
    (848, 480, 10, 848, 480, 10),
)


class RealsenseCamera:
    """RGB stream from a single RealSense device.

    Depth is enabled as well so downstream code can fuse it later; only RGB is
    exposed for now (ChArUco detection runs on the color stream).
    """

    def __init__(self, serial: str | None = None, config: StreamConfig = StreamConfig()):
        rs = _realsense()
        self._rs = rs
        self._ctx = rs.context()
        self._device = self._select_device(serial)
        self._pipeline = rs.pipeline(self._ctx)
        requested = (config.width, config.height, config.fps,
                     config.width, config.height, config.fps)
        candidates = [requested] + [c for c in _FALLBACKS if c != requested]
        last_error: RuntimeError | None = None
        for cw, ch, cfps, dw, dh, dfps in candidates:
            pipeline_config = rs.config()
            pipeline_config.enable_stream(rs.stream.color, cw, ch, rs.format.rgb8, cfps)
            pipeline_config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, dfps)
            try:
                self._pipeline.start(pipeline_config)
            except RuntimeError as exc:
                last_error = exc
                continue
            self.stream_info = f"color {cw}x{ch}@{cfps} + depth {dw}x{dh}@{dfps}"
            return
        raise CameraError(
            f"No stream profile resolved on this connection (last error: {last_error}); "
            "USB 2.0 ports limit the D435i, try a USB 3.0 port or lower --width/--height")

    def _select_device(self, serial: str | None):
        devices = self._ctx.query_devices()
        if not devices:
            raise CameraError(
                "No RealSense device found; connect the D435i, check the USB cable "
                "(USB 3.0) and that no other process holds the camera")
        serial_of = lambda d: d.get_info(rs.camera_info.serial_number)
        if serial is None:
            if len(devices) > 1:
                found = [serial_of(d) for d in devices]
                raise CameraError(f"{len(devices)} RealSense devices connected; select one with --serial {found}")
            return devices[0]
        for device in devices:
            if serial_of(device) == serial:
                return device
        available = [serial_of(d) for d in devices]
        raise CameraError(f"RealSense serial {serial!r} not found; available: {available}")

    @property
    def intrinsics(self) -> dict:
        profile = self._pipeline.get_active_profile().get_stream(self._rs.stream.color)
        intr = profile.as_video_stream_profile().get_intrinsics()
        return {"fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy,
                "width": intr.width, "height": intr.height}

    def read(self) -> Frame:
        frames = self._pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            raise CameraError("No color frame from RealSense pipeline")
        bgr = np.ascontiguousarray(color.get_data())[:, :, ::-1]  # rgb8 -> BGR
        return Frame(
            bgr=bgr,
            intrinsics=self.intrinsics,
            timestamp_ms=float(color.get_timestamp()),
        )

    def close(self) -> None:
        self._pipeline.stop()

    def __enter__(self) -> "RealsenseCamera":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
