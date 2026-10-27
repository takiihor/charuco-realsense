from types import SimpleNamespace as NS
import threading

import numpy as np
import pytest

from charuco_realsense import camera as module
from charuco_realsense.camera import CameraCancelled, CameraError, RealsenseCamera, StreamConfig


@pytest.fixture
def sdk(monkeypatch):
    class Sensor:
        values = {"enable_auto_exposure": 1., "exposure": 100., "gain": 5.,
                  "enable_auto_white_balance": 1., "white_balance": 4500.}

        def __init__(self):
            self.values = dict(self.values)

        def get_stream_profiles(self):
            return [NS(stream_type=lambda: "color")]

        def supports(self, option):
            return option in self.values

        def get_option(self, option):
            return self.values[option]

        def set_option(self, option, value):
            self.values[option] = value

    class Device:
        def __init__(self, serial):
            self.serial = serial
            self.sensor = Sensor()

        def supports(self, option):
            return True

        def get_info(self, option):
            return self.serial if option == "serial_number" else "test"

        def query_sensors(self):
            return [self.sensor]

    devices = [Device("first"), Device("second")]

    class Config:
        def __init__(self):
            self.serial = None
            self.streams = []

        def enable_device(self, serial):
            self.serial = serial

        def enable_stream(self, *profile):
            self.streams.append(profile)

    def video(profile):
        stream, width, height, _, fps = profile
        intr = NS(fx=900., fy=901., ppx=width / 2, ppy=height / 2, width=width, height=height,
                  model="distortion.inverse_brown_conrady", coeffs=[0.] * 5)
        result = NS(get_intrinsics=lambda: intr, fps=lambda: fps,
                    width=lambda: width, height=lambda: height)
        result.as_video_stream_profile = lambda: result
        return result

    class Pipeline:
        def __init__(self, context):
            self.attempts = []
            self.stopped = False
            self.read_error = False

        def start(self, config):
            self.attempts.append(config)
            if config.streams[0][4] == 30 and config.streams[0][1] == 1280:
                raise RuntimeError("USB profile unsupported")
            profiles = {p[0]: video(p) for p in config.streams}
            return NS(get_device=lambda: next(d for d in devices if d.serial == config.serial),
                      get_stream=lambda stream: profiles[stream])

        def try_wait_for_frames(self, timeout_ms):
            if self.read_error:
                raise RuntimeError("disconnected")
            pixels = np.zeros((720, 1280, 3), np.uint8)
            pixels[:, :, :] = [10, 20, 30]
            color = NS(get_data=lambda: pixels, get_timestamp=lambda: 123., get_frame_number=lambda: 7,
                       get_frame_timestamp_domain=lambda: "hardware_clock")
            return True, NS(get_color_frame=lambda: color)

        def stop(self):
            self.stopped = True

    fake = NS(context=lambda: NS(query_devices=lambda: devices), pipeline=Pipeline, config=Config,
              stream=NS(color="color", depth="depth"), format=NS(rgb8="rgb8", z16="z16"),
              camera_info=NS(**{key: key for key in ("name", "serial_number", "firmware_version",
                                                    "physical_port", "usb_type_descriptor")}),
              option=NS(**{key: key for key in Sensor.values}), devices=devices)
    monkeypatch.setattr(module, "_realsense", lambda: fake)
    return fake


def test_serial_bound_on_every_fallback_and_actual_calibration(sdk):
    with RealsenseCamera("second") as camera:
        assert all(c.serial == "second" for c in camera._pipeline.attempts)
        assert len(camera._pipeline.attempts) == 2
        assert camera.actual_config == {"width": 1280, "height": 720, "fps": 15, "depth": False}
        assert all(len(c.streams) == 1 for c in camera._pipeline.attempts)
        assert camera.intrinsics["model"] == "inverse_brown_conrady"
        assert camera.intrinsics["coeffs"] == [0.] * 5
        assert camera.metadata["device"]["serial_number"] == "second"
        frame = camera.read()
        assert frame.frame_number == 7 and frame.timestamp_ms == 123.
        assert frame.bgr.flags.c_contiguous
        np.testing.assert_array_equal(frame.bgr[0, 0], [30, 20, 10])
        pipeline = camera._pipeline
    assert pipeline.stopped
    camera.close()  # Idempotent cleanup.


def test_multiple_missing_and_implicit_serials(sdk):
    with pytest.raises(CameraError, match="Multiple"):
        RealsenseCamera()
    with pytest.raises(CameraError, match="not found"):
        RealsenseCamera("missing")
    sdk.devices.pop()
    with RealsenseCamera() as camera:
        assert camera.serial == "first"
    sdk.devices.clear()
    with pytest.raises(CameraError, match="No RealSense"):
        RealsenseCamera()


def test_optional_depth_and_exposure_restoration(sdk):
    with RealsenseCamera("first", StreamConfig(depth=True)) as camera:
        assert len(camera._pipeline.attempts[-1].streams) == 2
        assert camera.actual_config["depth_profile"]["fps"] == 15
        camera.lock_exposure()
        assert sdk.devices[0].sensor.values["enable_auto_exposure"] == 0.
    assert sdk.devices[0].sensor.values["enable_auto_exposure"] == 1.


def test_read_errors_become_camera_errors_and_release_device(sdk):
    with pytest.raises(CameraError, match="disconnected"):
        with RealsenseCamera("first") as camera:
            camera._pipeline.read_error = True
            pipeline = camera._pipeline
            camera.read()
    assert pipeline.stopped


def test_cancel_waiting_for_frames_releases_camera(sdk):
    cancelled = threading.Event()
    with RealsenseCamera("first", stop_event=cancelled) as camera:
        calls = []

        def no_frame(timeout_ms):
            calls.append(timeout_ms)
            cancelled.set()
            return False, None
        camera._pipeline.try_wait_for_frames = no_frame
        with pytest.raises(CameraCancelled):
            camera.read()
        assert calls == [100]
        pipeline = camera._pipeline
    assert pipeline.stopped


def test_cancel_connection_skips_fallback_profiles(sdk):
    cancelled = threading.Event()
    pipelines = []
    original_pipeline = sdk.pipeline

    def pipeline(context):
        result = original_pipeline(context)
        original_start = result.start

        def start(config):
            cancelled.set()
            return original_start(config)  # The first profile fails as usual.
        result.start = start
        pipelines.append(result)
        return result
    sdk.pipeline = pipeline
    with pytest.raises(CameraCancelled):
        RealsenseCamera("first", stop_event=cancelled)
    assert len(pipelines[0].attempts) == 1


def test_frame_timeout_is_bounded(sdk, monkeypatch):
    with RealsenseCamera("first") as camera:
        clock = iter([0., 6.])
        monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
        camera._pipeline.try_wait_for_frames = lambda timeout_ms: (False, None)
        with pytest.raises(CameraError, match="5 seconds"):
            camera.read()
