from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def save_decoded_frame(outdir: Path, frame_id: int, raw_view_img: np.ndarray) -> None:
    """Save the rotated raw frame as a lossless PNG."""
    stem = f"frame_{frame_id:06d}"
    cv2.imwrite(str(outdir / f"{stem}_raw_rotated.png"), raw_view_img)


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
