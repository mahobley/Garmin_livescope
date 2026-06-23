#!/usr/bin/env python3
"""
Live Garmin GLS 10 / LiveScope JPEG viewer.

What it does:
  - Listens for Garmin UDP chunk packets on a network interface.
  - Reassembles chunks from stream 0x00060044 into frames.
  - Extracts the embedded JPEG.
  - Displays it live with OpenCV.
  - Optional: saves JPEG frames to disk.
  - Optional: records the displayed view to video.

Install:
  pip install scapy opencv-python numpy

Run:
  sudo python3 garmin_livescope_live_viewer.py --iface en0

Notes:
  - On macOS/Linux, packet sniffing usually requires sudo.
  - Use a managed switch with port mirroring between GLS 10 and chartplotter.
  - If the stream id differs, try --stream all first.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from scapy.all import sniff, UDP, Raw

from warp_garmin_polar_to_xy import COLOR_SCHEMES, apply_color_scheme, bilinear_sample


JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"
WINDOW_NAME = "Garmin GLS 10 LiveScope"
ECHOGRAM_WINDOW_NAME = "Garmin LiveScope Echogram"
RECORD_BUTTON = (10, 42, 126, 78)
VIEW_BUTTON = (136, 42, 252, 78)
MOTION_BUTTON = (262, 42, 390, 78)
GAIN_DOWN_BUTTON = (10, 86, 48, 122)
GAIN_VALUE_RECT = (48, 86, 112, 122)
GAIN_UP_BUTTON = (112, 86, 150, 122)
MOTION_GAIN_STEP = 0.5


@dataclass
class ChunkHeader:
    magic: int
    len_after8: int
    stream_id: int
    frame_id: int
    total_len: int
    offset: int
    chunk_len: int


def parse_chunk_header(payload: bytes) -> Optional[ChunkHeader]:
    if len(payload) < 28:
        return None

    vals = np.frombuffer(payload[:28], dtype="<u4", count=7)
    h = ChunkHeader(*map(int, vals))

    if h.magic != 0x00000904:
        return None
    if h.total_len <= 0 or h.chunk_len <= 0:
        return None
    if h.chunk_len > len(payload) - 28:
        return None
    if h.offset < 0 or h.offset + h.chunk_len > h.total_len + 4096:
        return None

    return h


def find_jpeg_span(data: bytes) -> Optional[Tuple[int, int]]:
    start = data.find(JPEG_SOI)
    if start < 0:
        return None

    end = data.find(JPEG_EOI, start + 2)
    if end < 0:
        return None

    return start, end + 2


def read_theta_table_from_prejpg(prejpg: bytes, n_theta: int, offset: int = 68) -> np.ndarray:
    """Read n_theta little-endian float32 theta values from the pre-JPEG bytes."""
    need = offset + n_theta * 4
    if len(prejpg) < need:
        raise ValueError(f"pre-JPEG block is too short for {n_theta} theta floats at offset {offset}")

    theta = np.frombuffer(prejpg[offset:need], dtype="<f4").astype(np.float64)
    diffs = np.diff(theta)
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        raise ValueError("theta table is not monotonic")
    if np.nanmax(np.abs(theta)) > math.pi * 1.25:
        raise ValueError("theta table does not look like radians")
    return theta


def polar_image_to_xy(
    img: np.ndarray,
    prejpg: bytes,
    out_width: int = 900,
    out_height: int = 900,
    max_range: Optional[float] = None,
    range_offset_bins: float = 0.0,
    theta_offset_deg: float = 0.0,
    flip_theta: bool = False,
    flip_range: bool = False,
    forward_is_up: bool = True,
) -> np.ndarray:
    """Warp one decoded grayscale Garmin frame from theta/range raster to X/Y raster."""
    n_theta, n_range = img.shape

    theta = read_theta_table_from_prejpg(prejpg, n_theta=n_theta)
    if flip_theta:
        theta = theta[::-1]
        img = img[::-1, :]
    if flip_range:
        img = img[:, ::-1]

    theta = theta + math.radians(theta_offset_deg)

    if max_range is None:
        r_max = float(n_range - 1 - range_offset_bins)
    else:
        r_max = float(max_range)
    r_min = 0.0

    theta_min = float(np.min(theta))
    theta_max = float(np.max(theta))
    x_extent = r_max * max(abs(math.sin(theta_min)), abs(math.sin(theta_max)))
    y_extent = r_max

    x = np.linspace(-x_extent, x_extent, out_width)
    if forward_is_up:
        y = np.linspace(y_extent, 0.0, out_height)
    else:
        y = np.linspace(0.0, y_extent, out_height)
    X, Y = np.meshgrid(x, y)

    R = np.sqrt(X * X + Y * Y)
    TH = np.arctan2(X, Y)

    row_indices = np.arange(n_theta, dtype=np.float64)
    if theta[0] < theta[-1]:
        src_row = np.interp(TH, theta, row_indices, left=np.nan, right=np.nan)
    else:
        src_row = np.interp(TH, theta[::-1], row_indices[::-1], left=np.nan, right=np.nan)

    if max_range is None:
        src_col = R + range_offset_bins
    else:
        src_col = (R / max_range) * (n_range - 1 - range_offset_bins) + range_offset_bins

    valid_sector = (
        np.isfinite(src_row)
        & (R >= r_min)
        & (R <= r_max)
        & (TH >= theta_min)
        & (TH <= theta_max)
    )
    src_row = np.where(valid_sector, src_row, -1.0)
    src_col = np.where(valid_sector, src_col, -1.0)

    return bilinear_sample(img, src_row, src_col, fill=0)


def colorize_for_cv2(img: np.ndarray, color_scheme: str) -> np.ndarray:
    """Convert an 8-bit grayscale image into an OpenCV-displayable BGR image."""
    pil_img = apply_color_scheme(img, color_scheme)
    arr = np.asarray(pil_img)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def rotate_raw_view(img: np.ndarray) -> np.ndarray:
    """Rotate the decoded raw Garmin frame into the preferred viewing orientation."""
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)


def make_video_writer(path: Path, fps: float, frame_size: Tuple[int, int]) -> cv2.VideoWriter:
    """Create an OpenCV video writer for the displayed BGR frames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".mp4", ".m4v"}:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    elif suffix == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")
    return writer


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
    writer: Optional[cv2.VideoWriter] = None
    active_path: Optional[Path] = None
    frame_size: Optional[Tuple[int, int]] = None
    clip_index: int = 0
    requested: bool = False

    def toggle_requested(self) -> None:
        if self.writer is not None:
            self.requested = False
            self.stop()
            return
        self.requested = not self.requested

    def sync(self, frame_size: Tuple[int, int]) -> None:
        if self.requested and self.writer is None:
            self.clip_index += 1
            self.active_path = video_clip_path(self.base_path, self.clip_index)
            self.frame_size = frame_size
            self.writer = make_video_writer(self.active_path, self.fps, frame_size)
            print(f"Recording {self.label} to {self.active_path}")
        elif not self.requested and self.writer is not None:
            self.stop()

    def write(self, frame: np.ndarray) -> None:
        if self.writer is not None:
            if self.frame_size is not None:
                width, height = self.frame_size
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            self.writer.write(frame)

    def stop(self) -> None:
        if self.writer is None:
            return
        self.writer.release()
        self.writer = None
        self.frame_size = None
        print(f"Video saved: {self.active_path}")


@dataclass
class EchogramState:
    width: int
    height: int
    mode: str = "max"
    image: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.width = max(1, self.width)
        self.height = max(1, self.height)
        self.image = np.zeros((self.height, self.width), dtype=np.uint8)

    def add_frame(self, polar_img: np.ndarray) -> np.ndarray:
        """Append one time column derived from the frame's range profile."""
        if self.image is None:
            self.image = np.zeros((self.height, self.width), dtype=np.uint8)

        if self.mode == "mean":
            profile = np.mean(polar_img, axis=0).astype(np.uint8)
        else:
            profile = np.max(polar_img, axis=0).astype(np.uint8)

        column = cv2.resize(profile[:, None], (1, self.height), interpolation=cv2.INTER_AREA)[:, 0]
        self.image[:, :-1] = self.image[:, 1:]
        self.image[:, -1] = column
        return self.image.copy()


@dataclass
class ViewState:
    warp_enabled: bool

    def toggle(self) -> None:
        self.warp_enabled = not self.warp_enabled
        mode = "warped X/Y" if self.warp_enabled else "raw Garmin JPEG"
        print(f"Display mode: {mode}")


@dataclass
class MotionState:
    enabled: bool
    alpha: float
    threshold: int
    gain: float
    background: Optional[np.ndarray] = None

    def toggle(self) -> None:
        self.enabled = not self.enabled
        self.background = None
        mode = "on" if self.enabled else "off"
        print(f"Motion filter: {mode}")

    def adjust_gain(self, delta: float) -> None:
        self.gain = max(0.0, self.gain + delta)
        print(f"Motion gain: {self.gain:.1f}")

    def set_gain(self, gain: float) -> None:
        self.gain = max(0.0, gain)
        print(f"Motion gain: {self.gain:.1f}")

    def apply(self, img: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return img

        current = img.astype(np.float32)
        if self.background is None or self.background.shape != current.shape:
            self.background = current.copy()
            return np.zeros_like(img)

        diff = cv2.absdiff(current, self.background)
        cv2.accumulateWeighted(current, self.background, self.alpha)

        motion = np.maximum(diff - float(self.threshold), 0.0) * self.gain
        return np.clip(motion, 0, 255).astype(np.uint8)


def put_centered_text(
    display: np.ndarray,
    text: str,
    rect: Tuple[int, int, int, int],
    font_scale: float,
    thickness: int = 2,
) -> None:
    """Draw text centered inside a rectangle."""
    x1, y1, x2, y2 = rect
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x = x1 + max(0, (x2 - x1 - text_w) // 2)
    y = y1 + max(text_h, (y2 - y1 + text_h) // 2) - baseline // 2
    cv2.putText(
        display,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
    )


def draw_record_button(display: np.ndarray, recording_enabled: bool, recording: bool) -> None:
    """Draw a simple clickable recording button onto the OpenCV frame."""
    if not recording_enabled:
        return

    x1, y1, x2, y2 = RECORD_BUTTON
    fill = (32, 32, 180) if recording else (40, 120, 40)
    label = "STOP REC" if recording else "START REC"

    cv2.rectangle(display, (x1, y1), (x2, y2), fill, thickness=-1)
    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 255), thickness=1)
    put_centered_text(display, label, RECORD_BUTTON, font_scale=0.48)


def draw_motion_button(display: np.ndarray, motion_enabled: bool) -> None:
    """Draw a clickable motion/background subtraction toggle."""
    x1, y1, x2, y2 = MOTION_BUTTON
    fill = (32, 90, 150) if motion_enabled else (80, 80, 80)
    label = "MOTION OFF" if motion_enabled else "MOTION ON"

    cv2.rectangle(display, (x1, y1), (x2, y2), fill, thickness=-1)
    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 255), thickness=1)
    put_centered_text(display, label, MOTION_BUTTON, font_scale=0.45)


def draw_gain_buttons(display: np.ndarray, gain: float) -> None:
    """Draw +/- controls for the motion gain."""
    for rect, label in ((GAIN_DOWN_BUTTON, "-"), (GAIN_UP_BUTTON, "+")):
        x1, y1, x2, y2 = rect
        cv2.rectangle(display, (x1, y1), (x2, y2), (70, 70, 70), thickness=-1)
        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 255), thickness=1)
        put_centered_text(display, label, rect, font_scale=0.75)

    put_centered_text(display, f"{gain:.1f}", GAIN_VALUE_RECT, font_scale=0.55)


def prompt_motion_gain(motion: MotionState) -> None:
    """Prompt in the terminal for an exact motion gain value."""
    try:
        value = input(f"Enter motion gain [{motion.gain:.1f}]: ").strip()
    except EOFError:
        print("Motion gain unchanged.")
        return

    if not value:
        print("Motion gain unchanged.")
        return

    try:
        motion.set_gain(float(value))
    except ValueError:
        print(f"Invalid motion gain: {value!r}")


def draw_view_button(display: np.ndarray, warp_enabled: bool) -> None:
    """Draw a clickable raw/warped view toggle onto the OpenCV frame."""
    x1, y1, x2, y2 = VIEW_BUTTON
    fill = (110, 70, 30) if warp_enabled else (80, 80, 80)
    label = "RAW VIEW" if warp_enabled else "WARP VIEW"

    cv2.rectangle(display, (x1, y1), (x2, y2), fill, thickness=-1)
    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 255), thickness=1)
    put_centered_text(display, label, VIEW_BUTTON, font_scale=0.48)


class GarminFrameAssembler:
    def __init__(self, stream_filter: Optional[int] = 0x00060044):
        self.stream_filter = stream_filter
        self.parts: Dict[Tuple[int, int, int], Dict[int, bytes]] = {}
        self.last_cleanup = time.time()

    def add_payload(self, payload: bytes) -> Optional[Tuple[int, bytes, bytes]]:
        h = parse_chunk_header(payload)
        if h is None:
            return None

        if self.stream_filter is not None and h.stream_id != self.stream_filter:
            return None

        key = (h.stream_id, h.frame_id, h.total_len)
        self.parts.setdefault(key, {})[h.offset] = payload[28:28 + h.chunk_len]

        # Fast completeness check
        received = sum(len(v) for v in self.parts[key].values())
        if received < h.total_len:
            return None

        buf = bytearray(h.total_len)
        mask = np.zeros(h.total_len, dtype=np.bool_)

        for offset, chunk in self.parts[key].items():
            end = min(h.total_len, offset + len(chunk))
            if 0 <= offset < h.total_len:
                buf[offset:end] = chunk[:end - offset]
                mask[offset:end] = True

        if not bool(mask.all()):
            return None

        del self.parts[key]
        frame_data = bytes(buf)
        span = find_jpeg_span(frame_data)
        if span is None:
            return None
        start, end = span
        prejpg = frame_data[:start]
        jpg = frame_data[start:end]

        self.cleanup_old_frames(h.frame_id)
        return h.frame_id, jpg, prejpg

    def cleanup_old_frames(self, newest_frame_id: int) -> None:
        now = time.time()
        if now - self.last_cleanup < 2:
            return

        old_keys = [
            key for key in self.parts
            if key[1] < newest_frame_id - 30
        ]
        for key in old_keys:
            del self.parts[key]

        self.last_cleanup = now


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iface", required=True, help="Network interface, e.g. en0, eth0, Ethernet")
    parser.add_argument("--stream", default="0x00060044", help="Stream ID hex, or 'all'")
    parser.add_argument("--save", type=Path, default=None, help="Optional folder to save JPEG frames")
    parser.add_argument("--record-video", type=Path, default=Path("livescope.mp4"), help="Output path for the START REC button; default: livescope.mp4")
    parser.add_argument("--record-echogram", type=Path, default=Path("echogram.mp4"), help="Output path for the echogram START REC button; default: echogram.mp4")
    parser.add_argument("--video-fps", type=float, default=20.0, help="FPS to write into recorded video; default: 20")
    parser.add_argument("--udp-port", type=int, default=None, help="Optional UDP port filter")
    parser.add_argument("--warp-xy", action="store_true", help="Start in warped theta/range X/Y fan view")
    parser.add_argument("--width", type=int, default=900, help="warped output width when --warp-xy is enabled")
    parser.add_argument("--height", type=int, default=900, help="warped output height when --warp-xy is enabled")
    parser.add_argument("--max-range", type=float, default=None, help="real-world max range; otherwise range units are source bins")
    parser.add_argument("--range-offset-bins", type=float, default=0.0, help="add this source-column offset to the range mapping")
    parser.add_argument("--theta-offset-deg", type=float, default=0.0, help="rotate theta table by this many degrees")
    parser.add_argument("--flip-theta", action="store_true", help="reverse theta rows if left/right is backwards")
    parser.add_argument("--flip-range", action="store_true", help="reverse range columns if near/far is backwards")
    parser.add_argument("--forward-down", action="store_true", help="put forward/far range at bottom instead of top")
    parser.add_argument("--motion", action="store_true", help="Start with background-subtraction motion view enabled")
    parser.add_argument("--motion-alpha", type=float, default=0.04, help="Background learning rate for motion view; default: 0.04")
    parser.add_argument("--motion-threshold", type=int, default=14, help="Minimum brightness change shown in motion view; default: 14")
    parser.add_argument("--motion-gain", type=float, default=4.0, help="Brightness gain applied to motion differences; default: 4.0")
    parser.add_argument("--no-echogram", action="store_true", help="Disable the second scrolling echogram window")
    parser.add_argument("--echogram-width", type=int, default=700, help="Scrolling echogram window width; default: 700")
    parser.add_argument("--echogram-height", type=int, default=512, help="Scrolling echogram window height; default: 512")
    parser.add_argument("--echogram-mode", choices=("max", "mean"), default="max", help="Range profile reducer for echogram columns; default: max")
    parser.add_argument("--echogram-motion", action="store_true", help="Start echogram window with background subtraction enabled")
    parser.add_argument("--echogram-motion-alpha", type=float, default=0.04, help="Echogram background learning rate; default: 0.04")
    parser.add_argument("--echogram-motion-threshold", type=int, default=14, help="Minimum echogram brightness change shown; default: 14")
    parser.add_argument("--echogram-motion-gain", type=float, default=4.0, help="Brightness gain for echogram differences; default: 4.0")
    parser.add_argument(
        "--colorscheme",
        "--colourscheme",
        "--color-scheme",
        "--colour-scheme",
        default="orange",
        choices=sorted(COLOR_SCHEMES),
        help="warped display color ramp; default: orange",
    )
    args = parser.parse_args()

    stream_filter = None if args.stream.lower() == "all" else int(args.stream, 0)
    assembler = GarminFrameAssembler(stream_filter=stream_filter)

    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    t0 = time.time()
    recording = RecordingState(args.record_video, args.video_fps, label="raw footage")
    echogram_recording = RecordingState(args.record_echogram, args.video_fps, label="echogram footage")
    view = ViewState(warp_enabled=args.warp_xy)
    echogram = None if args.no_echogram else EchogramState(
        width=args.echogram_width,
        height=args.echogram_height,
        mode=args.echogram_mode,
    )
    motion = MotionState(
        enabled=args.motion,
        alpha=float(np.clip(args.motion_alpha, 0.0, 1.0)),
        threshold=max(0, args.motion_threshold),
        gain=max(0.0, args.motion_gain),
    )
    echogram_motion = MotionState(
        enabled=args.echogram_motion,
        alpha=float(np.clip(args.echogram_motion_alpha, 0.0, 1.0)),
        threshold=max(0, args.echogram_motion_threshold),
        gain=max(0.0, args.echogram_motion_gain),
    )

    print("Listening for Garmin LiveScope packets...")
    print("Press q in the OpenCV window to quit.")
    print("Click WARP VIEW/RAW VIEW in the OpenCV window, or press w, to switch display modes.")
    print("Click MOTION ON/OFF in the OpenCV window, or press m, to toggle background subtraction.")
    print("Click -/+/gain value in the OpenCV window, or press [, ], and g, to adjust motion gain.")
    print("Click START REC in the main OpenCV window, or press r, to start/stop raw recording.")
    if echogram is not None:
        print("Showing scrolling echogram in a second OpenCV window. Its START REC button records echogram footage.")
        print("Click MOTION ON/OFF in the echogram window, or press b, to toggle echogram background subtraction.")

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        x1, y1, x2, y2 = VIEW_BUTTON
        if x1 <= x <= x2 and y1 <= y <= y2:
            view.toggle()
            return

        x1, y1, x2, y2 = MOTION_BUTTON
        if x1 <= x <= x2 and y1 <= y <= y2:
            motion.toggle()
            return

        x1, y1, x2, y2 = GAIN_DOWN_BUTTON
        if x1 <= x <= x2 and y1 <= y <= y2:
            motion.adjust_gain(-MOTION_GAIN_STEP)
            return

        x1, y1, x2, y2 = GAIN_VALUE_RECT
        if x1 <= x <= x2 and y1 <= y <= y2:
            prompt_motion_gain(motion)
            return

        x1, y1, x2, y2 = GAIN_UP_BUTTON
        if x1 <= x <= x2 and y1 <= y <= y2:
            motion.adjust_gain(MOTION_GAIN_STEP)
            return

        if recording is not None:
            x1, y1, x2, y2 = RECORD_BUTTON
            if x1 <= x <= x2 and y1 <= y <= y2:
                recording.toggle_requested()

    def on_echogram_mouse(event, x, y, _flags, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        x1, y1, x2, y2 = RECORD_BUTTON
        if x1 <= x <= x2 and y1 <= y <= y2:
            echogram_recording.toggle_requested()
            return

        x1, y1, x2, y2 = MOTION_BUTTON
        if x1 <= x <= x2 and y1 <= y <= y2:
            echogram_motion.toggle()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    cv2.moveWindow(WINDOW_NAME, 20, 40)
    if echogram is not None:
        cv2.namedWindow(ECHOGRAM_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(ECHOGRAM_WINDOW_NAME, on_echogram_mouse)
        cv2.resizeWindow(ECHOGRAM_WINDOW_NAME, echogram.width, echogram.height)
        cv2.moveWindow(ECHOGRAM_WINDOW_NAME, 560, 40)

    bpf = "udp"
    if args.udp_port is not None:
        bpf = f"udp port {args.udp_port}"

    def handle_packet(pkt) -> None:
        nonlocal frame_count, t0

        if UDP not in pkt or Raw not in pkt:
            return

        payload = bytes(pkt[Raw].load)
        result = assembler.add_payload(payload)
        if result is None:
            return

        frame_id, jpg, prejpg = result

        arr = np.frombuffer(jpg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return

        if echogram is not None:
            echogram_img = echogram.add_frame(img)
            echogram_img = echogram_motion.apply(echogram_img)
            echogram_display = cv2.cvtColor(echogram_img, cv2.COLOR_GRAY2BGR)
            echogram_recording.sync((echogram_display.shape[1], echogram_display.shape[0]))
            echogram_recording.write(echogram_display)
            draw_record_button(
                echogram_display,
                recording_enabled=True,
                recording=echogram_recording.writer is not None,
            )
            draw_motion_button(echogram_display, motion_enabled=echogram_motion.enabled)
            cv2.imshow(ECHOGRAM_WINDOW_NAME, echogram_display)

        raw_view_img = rotate_raw_view(img)
        raw_record_frame = cv2.cvtColor(raw_view_img, cv2.COLOR_GRAY2BGR)

        frame_count += 1
        elapsed = max(0.001, time.time() - t0)
        fps = frame_count / elapsed

        if view.warp_enabled:
            try:
                warped = polar_image_to_xy(
                    img,
                    prejpg,
                    out_width=args.width,
                    out_height=args.height,
                    max_range=args.max_range,
                    range_offset_bins=args.range_offset_bins,
                    theta_offset_deg=args.theta_offset_deg,
                    flip_theta=args.flip_theta,
                    flip_range=args.flip_range,
                    forward_is_up=not args.forward_down,
                )
            except ValueError as exc:
                print(f"frame {frame_id}: cannot warp: {exc}")
                return
            display_img = motion.apply(warped)
            display = colorize_for_cv2(display_img, args.colorscheme)
        else:
            display_img = motion.apply(raw_view_img)
            display = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

        cv2.putText(
            display,
            f"Frame {frame_id} | {fps:.1f} FPS",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        if recording:
            height, width = raw_record_frame.shape[:2]
            recording.sync((width, height))
            recording.write(raw_record_frame)

        if recording:
            draw_record_button(display, recording_enabled=True, recording=recording.writer is not None)
        draw_view_button(display, warp_enabled=view.warp_enabled)
        draw_motion_button(display, motion_enabled=motion.enabled)
        draw_gain_buttons(display, gain=motion.gain)

        cv2.imshow(WINDOW_NAME, display)

        if args.save:
            out = args.save / f"frame_{frame_id:06d}.jpg"
            cv2.imwrite(str(out), raw_view_img)
            if view.warp_enabled:
                warped_out = args.save / f"frame_{frame_id:06d}_xy.png"
                cv2.imwrite(str(warped_out), display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("r") and recording:
            recording.toggle_requested()
        if key == ord("e") and echogram is not None:
            echogram_recording.toggle_requested()
        if key == ord("b") and echogram is not None:
            echogram_motion.toggle()
        if key == ord("w"):
            view.toggle()
        if key == ord("m"):
            motion.toggle()
        if key == ord("["):
            motion.adjust_gain(-MOTION_GAIN_STEP)
        if key == ord("]"):
            motion.adjust_gain(MOTION_GAIN_STEP)
        if key == ord("g"):
            prompt_motion_gain(motion)
        if key == ord("q"):
            raise KeyboardInterrupt

    try:
        sniff(iface=args.iface, filter=bpf, prn=handle_packet, store=False)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if recording is not None:
            recording.stop()
        if echogram is not None:
            echogram_recording.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
