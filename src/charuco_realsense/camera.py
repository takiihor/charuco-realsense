"""Intel RealSense capture; the SDK is imported only when hardware is used."""
from dataclasses import dataclass
import re
import time

import numpy as np


CAMERA_MODELS = ("auto", "D435i", "D415")


class CameraError(RuntimeError):
    pass


class CameraCancelled(CameraError):
    """A requested shutdown, rather than a camera failure."""


@dataclass(frozen=True)
class StreamConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    depth: bool = False

    def __post_init__(self):
        if any(not isinstance(v, int) or v <= 0 for v in (self.width, self.height, self.fps)):
            raise ValueError("Stream width, height and fps must be positive integers")


@dataclass(frozen=True)
class Frame:
    bgr: np.ndarray
    intrinsics: dict
    timestamp_ms: float
    frame_number: int = 0
    timestamp_domain: str = "unknown"


def _realsense():
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise CameraError("pyrealsense2 is not installed; run: pip install pyrealsense2") from exc
    return rs


def _device_info(rs, device):
    result = {}
    for key in ("name", "serial_number", "firmware_version", "physical_port", "usb_type_descriptor"):
        option = getattr(rs.camera_info, key)
        if device.supports(option):
            result[{"firmware_version": "firmware", "usb_type_descriptor": "usb_type"}.get(key, key)] = device.get_info(option)
    return result


def list_devices() -> list[dict]:
    rs = _realsense()
    try:
        return [_device_info(rs, device) for device in rs.context().query_devices()]
    except RuntimeError as exc:
        raise CameraError(f"Cannot list RealSense devices: {exc}") from exc


def _matching_devices(rs, devices, serial=None, model="auto"):
    if model not in CAMERA_MODELS:
        raise ValueError(f"Unknown camera model {model!r}; choose from {CAMERA_MODELS}")
    if serial is not None:
        devices = [d for d in devices if d.get_info(rs.camera_info.serial_number) == serial]
    if model != "auto":
        devices = [d for d in devices if re.search(rf"\b{re.escape(model)}\b",
                                                  d.get_info(rs.camera_info.name), re.IGNORECASE)]
    return devices


def reset_device(serial: str | None = None, model: str = "auto") -> bool:
    """Hardware-reset an enumerable, idle camera to recover a wedged USB device.

    The device re-enumerates a few seconds later. Returns False if no matching
    device is visible; only a single device matching the serial/model is reset.
    """
    rs = _realsense()
    try:
        devices = _matching_devices(rs, list(rs.context().query_devices()), serial, model)
        if len(devices) != 1:
            return False
        devices[0].hardware_reset()
        return True
    except RuntimeError as exc:
        raise CameraError(f"Cannot reset RealSense device: {exc}") from exc


# color_w/h/fps, depth_w/h/fps (depth is optional).
_FALLBACKS = ((1280, 720, 15, 640, 480, 15),
              (640, 480, 30, 640, 480, 30),
              (848, 480, 10, 848, 480, 10))


class RealsenseCamera:
    """A selected device's color stream, with optional depth for future fusion."""

    def __init__(self, serial: str | None = None, config: StreamConfig = StreamConfig(), stop_event=None,
                 model: str = "auto"):
        self._rs = rs = _realsense()
        self._stop_event = stop_event
        self._started = False
        self._restore_options = []
        try:
            self._check_cancelled()
            self._ctx = rs.context()
            self._device = self._select_device(serial, model)
            self.serial = self._device.get_info(rs.camera_info.serial_number)
            self._connection_info = _device_info(rs, self._device)
            self._pipeline = rs.pipeline(self._ctx)
            requested = (config.width, config.height, config.fps,
                         config.width, config.height, config.fps)
            candidates = [requested] + [c for c in _FALLBACKS if c != requested]
            last_error = frame_error = None
            for cw, ch, cfps, dw, dh, dfps in candidates:
                self._check_cancelled()
                pipeline_config = rs.config()
                pipeline_config.enable_device(self.serial)
                pipeline_config.enable_stream(rs.stream.color, cw, ch, rs.format.rgb8, cfps)
                if config.depth:
                    pipeline_config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, dfps)
                try:
                    profile = self._pipeline.start(pipeline_config)
                except RuntimeError as exc:
                    last_error = exc
                    continue
                self._started = True
                self._check_cancelled()
                self._device = profile.get_device()
                color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
                intr = color_profile.get_intrinsics()
                self._intrinsics = {
                    "fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy,
                    "width": intr.width, "height": intr.height,
                    "model": str(intr.model).removeprefix("distortion."),
                    "coeffs": list(intr.coeffs),
                }
                self.actual_config = {"width": intr.width, "height": intr.height,
                                      "fps": color_profile.fps(), "depth": config.depth}
                self.stream_info = f"color {intr.width}x{intr.height}@{color_profile.fps()}"
                if config.depth:
                    depth = profile.get_stream(rs.stream.depth).as_video_stream_profile()
                    self.actual_config["depth_profile"] = {"width": depth.width(), "height": depth.height(), "fps": depth.fps()}
                    self.stream_info += f" + depth {depth.width()}x{depth.height()}@{depth.fps()}"
                try:
                    # Some devices accept a profile but never deliver frames.
                    # Verify delivery before treating startup as successful.
                    self.read()
                except CameraCancelled:
                    raise
                except CameraError as exc:
                    frame_error = exc
                    self.close()
                    continue
                return
            raise CameraError(f"No working stream profile (last error: {frame_error or last_error}); "
                              "try USB 3.0, a lower profile, or disabling --depth")
        except BaseException as exc:
            self.close()
            if isinstance(exc, RuntimeError) and not isinstance(exc, CameraError):
                raise CameraError(f"Cannot start RealSense camera: {exc}") from exc
            raise

    def _check_cancelled(self):
        if self._stop_event is not None and self._stop_event.is_set():
            raise CameraCancelled("Camera shutdown requested")

    def _select_device(self, serial: str | None, model: str):
        devices = list(self._ctx.query_devices())
        if not devices:
            raise CameraError("No RealSense device found; connect it and check the USB cable")
        available = [_device_info(self._rs, d) for d in devices]
        devices = _matching_devices(self._rs, devices, serial, model)
        if not devices:
            selection = f"RealSense {model}" if model != "auto" else "RealSense device"
            if serial is not None:
                selection += f" with serial {serial!r}"
            raise CameraError(f"{selection} not found; available: {available}")
        serials = [d.get_info(self._rs.camera_info.serial_number) for d in devices]
        if len(devices) > 1:
            raise CameraError(f"Multiple RealSense devices match model {model!r}; select --serial from {serials}")
        return devices[0]

    @property
    def intrinsics(self) -> dict:
        return {**self._intrinsics, "coeffs": list(self._intrinsics["coeffs"])}

    def _color_sensor(self):
        rs = self._rs
        for sensor in self._device.query_sensors():
            if any(p.stream_type() == rs.stream.color for p in sensor.get_stream_profiles()):
                return sensor
        raise CameraError("No color sensor available")

    def lock_exposure(self):
        """Freeze settled auto exposure/gain and restore auto mode on close."""
        rs = self._rs
        sensor = self._color_sensor()
        option = rs.option.enable_auto_exposure
        if not sensor.supports(option):
            raise CameraError("Color sensor does not support locking auto exposure")
        try:
            previous = sensor.get_option(option)
            manual = [(getattr(rs.option, name), sensor.get_option(getattr(rs.option, name)))
                      for name in ("exposure", "gain") if sensor.supports(getattr(rs.option, name))]
            sensor.set_option(option, 0.0)
            self._restore_options.append((sensor, option, previous))
            for manual_option, value in manual:
                self._restore_options.append((sensor, manual_option, value))
                sensor.set_option(manual_option, value)
        except RuntimeError as exc:
            raise CameraError(f"Cannot lock color exposure: {exc}") from exc

    @property
    def metadata(self) -> dict:
        rs = self._rs
        settings = {}
        try:
            sensor = self._color_sensor()
            for name in ("enable_auto_exposure", "exposure", "gain", "enable_auto_white_balance", "white_balance"):
                option = getattr(rs.option, name)
                if sensor.supports(option):
                    settings[name] = float(sensor.get_option(option))
            return {"device": _device_info(rs, self._device), "stream": self.actual_config,
                    "stream_info": self.stream_info, "intrinsics": self.intrinsics,
                    "color_settings": settings}
        except RuntimeError as exc:
            raise CameraError(f"Cannot read camera metadata: {exc}") from exc

    def read(self) -> Frame:
        try:
            # Short SDK waits let Python handle signals and the GUI cancel reads.
            # A missing camera still has the same overall five-second timeout.
            deadline = time.monotonic() + 5.
            while True:
                self._check_cancelled()
                available, frames = self._pipeline.try_wait_for_frames(timeout_ms=100)
                if available:
                    break
                if time.monotonic() >= deadline:
                    info = self._connection_info
                    name = info.get("name", "RealSense camera")
                    usb = info.get("usb_type", "unknown")
                    hint = ("Use a USB 3 data cable and USB 3 port without a hub or extension"
                            if usb.startswith("2") else "Check the USB cable and camera connection")
                    raise CameraError(f"No RealSense frames received for 5 seconds ({name}, USB {usb}); {hint}")
            self._check_cancelled()
            color = frames.get_color_frame()
            if not color:
                raise CameraError("No color frame from RealSense pipeline")
            # Copy owns the buffer and produces contiguous BGR data for OpenCV.
            bgr = np.ascontiguousarray(np.asanyarray(color.get_data())[:, :, ::-1])
            return Frame(bgr=bgr, intrinsics=self.intrinsics,
                         timestamp_ms=float(color.get_timestamp()),
                         frame_number=int(color.get_frame_number()),
                         timestamp_domain=str(color.get_frame_timestamp_domain()))
        except RuntimeError as exc:
            if isinstance(exc, CameraError):
                raise
            raise CameraError(f"Cannot read RealSense frame: {exc}") from exc

    def close(self) -> None:
        try:
            for sensor, option, value in reversed(self._restore_options):
                try:
                    sensor.set_option(option, value)
                except RuntimeError:
                    pass  # A disconnected device may no longer accept options.
            self._restore_options.clear()
        finally:
            if self._started:
                self._started = False
                try:
                    self._pipeline.stop()
                except RuntimeError:
                    pass

    def __enter__(self) -> "RealsenseCamera":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
