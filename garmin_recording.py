from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def make_video_writer(path: Path, fps: float, frame_size: Tuple[int, int]) -> cv2.VideoWriter:
    """Create an OpenCV video writer, preferring low-loss codecs when possible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".avi":
        codecs = ("FFV1", "HFYU", "MJPG")
    elif suffix in {".mkv"}:
        codecs = ("FFV1", "MJPG")
    elif suffix in {".mp4", ".m4v"}:
        codecs = ("avc1", "H264", "mp4v")
    else:
        codecs = ("FFV1", "MJPG", "mp4v")

    for codec in codecs:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, frame_size)
        if writer.isOpened():
            print(f"Video codec for {path}: {codec}")
            return writer

    raise RuntimeError(f"could not open video writer for {path} with codecs: {', '.join(codecs)}")


def video_clip_path(base_path: Path, clip_index: int) -> Path:
    """Return the path for a recording clip, preserving the exact first path."""
    if clip_index == 1:
        return base_path
    suffix = base_path.suffix or ".mp4"
    return base_path.with_name(f"{base_path.stem}_{clip_index:03d}{suffix}")


@dataclass
class RecordingState:
    base_path: Path
    fps: float
    label: str
    auto_segment_seconds: Optional[float] = None
    writer: Optional[cv2.VideoWriter] = None
    active_path: Optional[Path] = None
    frame_size: Optional[Tuple[int, int]] = None
    segment_started_at: Optional[float] = None
    next_video_frame_at: Optional[float] = None
    clip_index: int = 0
    requested: bool = False

    def toggle_requested(self) -> None:
        if self.writer is not None:
            self.requested = False
            self.stop()
            return
        self.requested = not self.requested

    def sync(self, frame_size: Tuple[int, int]) -> None:
        now = time.time()
        auto_active = self.auto_segment_seconds is not None and self.auto_segment_seconds > 0

        if auto_active and self.writer is not None and self.segment_started_at is not None:
            if now - self.segment_started_at >= self.auto_segment_seconds:
                self.stop()

        if (self.requested or auto_active) and self.writer is None:
            self.start(frame_size, now)
        elif not self.requested and not auto_active and self.writer is not None:
            self.stop()

    def start(self, frame_size: Tuple[int, int], started_at: Optional[float] = None) -> None:
        self.clip_index += 1
        self.active_path = video_clip_path(self.base_path, self.clip_index)
        self.frame_size = frame_size
        self.segment_started_at = time.time() if started_at is None else started_at
        self.next_video_frame_at = self.segment_started_at
        self.writer = make_video_writer(self.active_path, self.fps, frame_size)
        print(f"Recording {self.label} to {self.active_path}")

    def write(self, frame: np.ndarray) -> None:
        if self.writer is None:
            return

        if self.frame_size is not None:
            width, height = self.frame_size
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        now = time.time()
        frame_interval = 1.0 / max(self.fps, 0.001)
        if self.next_video_frame_at is None:
            self.next_video_frame_at = now

        # OpenCV writes constant-FPS video. Fill the timeline with repeats so
        # playback duration tracks real capture time when incoming FPS is lower.
        max_fill_frames = max(1, int(self.fps * 2))
        written = 0
        while self.next_video_frame_at <= now and written < max_fill_frames:
            self.writer.write(frame)
            self.next_video_frame_at += frame_interval
            written += 1

        if self.next_video_frame_at < now:
            self.next_video_frame_at = now + frame_interval

    def stop(self) -> None:
        if self.writer is None:
            return
        self.writer.release()
        self.writer = None
        self.frame_size = None
        self.segment_started_at = None
        self.next_video_frame_at = None
        print(f"Video saved: {self.active_path}")
