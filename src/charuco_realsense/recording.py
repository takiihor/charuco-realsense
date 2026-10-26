"""Durable per-frame JSONL recordings and summaries, including interrupted runs."""
from datetime import datetime, timezone
import json
from pathlib import Path

from .metrics import SessionStatistics


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


class SessionRecorder:
    def __init__(self, path: str, metadata: dict, save_frames: bool = False):
        self.path = Path(path)
        # A previous experiment must never be silently overwritten or mixed in.
        self.path.mkdir(parents=True, exist_ok=False)
        self.metadata = {"schema_version": 1, **metadata}
        self.save_frames = save_frames
        if save_frames:
            (self.path / "images").mkdir()
        write_json(self.path / "session.json", self.metadata)
        self.statistics = SessionStatistics(metadata["board"]["expected_charuco_corner_count"],
                                            metadata.get("reference_pose"))
        self._file = (self.path / "frames.jsonl").open("w", encoding="utf-8")
        self.summary = None

    def add(self, record, image=None):
        if self.save_frames:
            import cv2
            relative = f"images/{record['frame_index']:08d}.png"
            if image is None or not cv2.imwrite(str(self.path / relative), image):
                raise OSError(f"Cannot save frame {relative}")
            record["image"] = relative
        encoded = json.dumps(record, sort_keys=True, allow_nan=False)
        self._file.write(encoded + "\n")
        self._file.flush()
        self.statistics.add(record)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._file.close()
        status = "complete" if exc_type is None else "interrupted" if issubclass(exc_type, KeyboardInterrupt) else "error"
        self.summary = {"schema_version": 1, "status": status,
                        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                        "error": str(exc) if exc is not None else None,
                        **self.statistics.summary()}
        write_json(self.path / "summary.json", self.summary)


def summarize_recording(path: str, reference_pose: dict | None = None) -> dict:
    directory = Path(path)
    metadata = json.loads((directory / "session.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported recording schema version")
    stats = SessionStatistics(metadata["board"]["expected_charuco_corner_count"],
                              reference_pose if reference_pose is not None else metadata.get("reference_pose"))
    with (directory / "frames.jsonl").open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                try:
                    stats.add(json.loads(line))
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"Invalid frame record on line {line_number}: {exc}") from exc
    return {"schema_version": 1, "status": "reanalyzed", **stats.summary()}
