from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from warp_garmin_polar_to_xy import COLOR_SCHEMES, apply_color_scheme, bilinear_sample


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


def prepare_raw_view_and_record_frame(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the rotated raw display frame and the BGR frame used for raw recording."""
    raw_view_img = rotate_raw_view(img)
    raw_record_frame = cv2.cvtColor(raw_view_img, cv2.COLOR_GRAY2BGR)
    return raw_view_img, raw_record_frame
