from __future__ import annotations

from datetime import datetime
import re
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


THETA_TABLE_OFFSET = 68
RANGE_SCALE_OFFSET = 32
RANGE_FEET_SCALE = 2.221279251
RANGE_FEET_OFFSET = 0.020227821
FEET_TO_METERS = 0.3048
CELSIUS_TO_FAHRENHEIT_SCALE = 9.0 / 5.0
CELSIUS_TO_FAHRENHEIT_OFFSET = 32.0


def save_decoded_frame(
    outdir: Path,
    frame_id: int,
    raw_view_img: np.ndarray,
    prejpg: bytes,
    polar_shape: tuple[int, int],
    capture_time: float,
    temperature_c: Optional[float],
) -> None:
    """Save the rotated raw frame and warp metadata."""
    stem = f"frame_{frame_id:06d}"
    cv2.imwrite(str(outdir / f"{stem}_raw_rotated.png"), raw_view_img)
    save_warp_metadata(outdir, stem, frame_id, prejpg, polar_shape, capture_time, temperature_c)


def save_warp_metadata(
    outdir: Path,
    stem: str,
    frame_id: int,
    prejpg: bytes,
    polar_shape: tuple[int, int],
    capture_time: float,
    temperature_c: Optional[float],
) -> None:
    """Write human-readable warp metadata beside the binary pre-JPEG block."""
    n_theta, n_range = polar_shape
    theta = read_theta_from_prejpg(prejpg, n_theta)
    range_scale = read_float32_from_prejpg(prejpg, RANGE_SCALE_OFFSET)
    selected_range_ft = estimate_selected_range_feet(range_scale)
    selected_range_m = feet_to_meters(selected_range_ft)
    captured_at = datetime.fromtimestamp(capture_time)
    temperature_f = celsius_to_fahrenheit(temperature_c)
    summary = [
        f"frame_id: {frame_id}",
        f"capture_unix_time: {capture_time:.6f}",
        f"capture_datetime_local: {captured_at.strftime('%Y-%m-%d %H:%M:%S.%f')}",
        f"capture_date_local: {captured_at.strftime('%Y-%m-%d')}",
        f"capture_time_local: {captured_at.strftime('%H:%M:%S.%f')}",
        f"temperature_celsius: {format_optional_float(temperature_c)}",
        f"temperature_fahrenheit: {format_optional_float(temperature_f)}",
        f"source_polar_rows_theta: {n_theta}",
        f"source_polar_columns_range: {n_range}",
        f"prejpg_bytes: {len(prejpg)}",
        f"range_scale_offset_bytes: {RANGE_SCALE_OFFSET}",
        f"range_scale_raw_float32: {format_optional_float(range_scale)}",
        f"estimated_selected_range_feet: {format_optional_float(selected_range_ft)}",
        f"estimated_selected_range_meters: {format_optional_float(selected_range_m)}",
        f"estimated_feet_per_range_column: {format_optional_float(selected_range_ft / n_range if selected_range_ft is not None else None)}",
        f"estimated_meters_per_range_column: {format_optional_float(selected_range_m / n_range if selected_range_m is not None else None)}",
        f"theta_table_offset_bytes: {THETA_TABLE_OFFSET}",
        "theta_units: degrees",
        f"theta_count: {0 if theta is None else len(theta)}",
    ]
    if theta is not None and len(theta) > 0:
        summary.extend(
            [
                f"theta_min_degrees: {float(np.degrees(np.min(theta))):.6f}",
                f"theta_max_degrees: {float(np.degrees(np.max(theta))):.6f}",
                f"theta_first_degrees: {float(np.degrees(theta[0])):.6f}",
                f"theta_last_degrees: {float(np.degrees(theta[-1])):.6f}",
                f"theta_csv: {stem}_theta.csv",
            ]
        )
        write_theta_csv(outdir / f"{stem}_theta.csv", theta)
    else:
        summary.append("theta_csv: unavailable")

    (outdir / f"{stem}_warp_metadata.txt").write_text("\n".join(summary) + "\n")


def read_theta_from_prejpg(prejpg: bytes, n_theta: int) -> Optional[np.ndarray]:
    need = THETA_TABLE_OFFSET + n_theta * 4
    if len(prejpg) < need:
        return None
    return np.frombuffer(prejpg[THETA_TABLE_OFFSET:need], dtype="<f4").astype(np.float64)


def read_float32_from_prejpg(prejpg: bytes, offset: int) -> Optional[float]:
    need = offset + 4
    if len(prejpg) < need:
        return None
    return float(np.frombuffer(prejpg[offset:need], dtype="<f4", count=1)[0])


def estimate_selected_range_feet(range_scale: Optional[float]) -> Optional[float]:
    if range_scale is None:
        return None
    return RANGE_FEET_SCALE * range_scale + RANGE_FEET_OFFSET


def feet_to_meters(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return value * FEET_TO_METERS


def celsius_to_fahrenheit(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return value * CELSIUS_TO_FAHRENHEIT_SCALE + CELSIUS_TO_FAHRENHEIT_OFFSET


def format_optional_float(value: Optional[float]) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.6f}"


def write_theta_csv(path: Path, theta: np.ndarray) -> None:
    lines = ["row,theta_degrees"]
    for row, value in enumerate(theta):
        lines.append(f"{row},{float(np.degrees(value)):.6f}")
    path.write_text("\n".join(lines) + "\n")


def choose_save_root(default: Path = Path("recordings")) -> Path:
    """Ask for a root output folder with a GUI dialog, falling back to terminal input."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(
            title="Choose where to save Garmin LiveScope files",
            initialdir=str(default.expanduser().resolve().parent),
            mustexist=False,
        )
        root.destroy()
        if selected:
            return Path(selected)
    except Exception as exc:
        print(f"Save-folder dialog unavailable: {exc}")

    entered = input(f"Save folder [{default}]: ").strip()
    return Path(entered) if entered else default


def under_save_root(path: Optional[Path], save_root: Path) -> Optional[Path]:
    """Resolve relative output paths under the selected save root."""
    if path is None or path.is_absolute():
        return path
    return save_root / path


def safe_name(name: str) -> str:
    """Make a filesystem-friendly session/file name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or time.strftime("livescope_%Y%m%d_%H%M%S")


def prompt_output_path(label: str, default: Path, is_dir: bool = False) -> Path:
    """Ask for one output file/folder name."""
    entered = input(f"{label} [{default}]: ").strip()
    if not entered:
        return default
    if is_dir:
        return Path(safe_name(entered))
    entered_path = Path(entered)
    suffix = default.suffix
    if entered_path.suffix == "" and suffix:
        entered_path = entered_path.with_suffix(suffix)
    return Path(safe_name(entered_path.name)) if entered_path.parent == Path(".") else entered_path


def append_timestamp_to_path(path: Path, timestamp: str) -> Path:
    """Append _YYYY-MM-DD_HHMMSS to a file or folder path."""
    if path.suffix:
        return path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
    return path.with_name(f"{path.name}_{timestamp}")


def unique_file(path: Path) -> Path:
    """Return path, or path_002.ext, path_003.ext, etc. if it already exists."""
    if not path.exists():
        return path
    suffix = path.suffix
    stem = path.stem
    for i in range(2, 10000):
        candidate = path.with_name(f"{stem}_{i:03d}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find unique file name for {path}")


def unique_dir(path: Path) -> Path:
    """Return path, or path_002, path_003, etc. if it already exists."""
    if not path.exists():
        return path
    for i in range(2, 10000):
        candidate = path.with_name(f"{path.name}_{i:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find unique folder name for {path}")
