#!/usr/bin/env python3
"""
Warp Garmin LiveScope-ish extracted JPEG frames from (theta, range) space to an X/Y image.

Assumption, based on the captures/scripts in this conversation:
  - Extracted JPEG is grayscale, shape roughly 511 rows x 512 cols.
  - JPEG rows are beams/angles (theta).
  - JPEG columns are range bins/distances.
  - The corresponding .prejpg.bin contains:
      68-byte metadata header
      511 little-endian float32 theta values at offset 68
      padding until the JPEG starts

This script works on:
  1) A single JPG plus its .prejpg.bin,
  2) A directory of frame_XXXXXX.jpg files with matching frame_XXXXXX.prejpg.bin files, or
  3) A saved live-viewer frame_XXXXXX_raw_rotated.png plus frame_XXXXXX_theta.csv.

Example:
  python warp_garmin_polar_to_xy.py garmin_out/frame_000123.jpg --pre garmin_out/frame_000123.prejpg.bin --out frame_000123_xy.png

Batch:
  python warp_garmin_polar_to_xy.py garmin_out --out xy_out --make-contact-sheet

Color schemes:
  orange  black -> red/orange -> yellow -> white, default
  gray    grayscale luminance
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


ColorStop = Tuple[float, Tuple[int, int, int]]


COLOR_SCHEMES: dict[str, Tuple[ColorStop, ...]] = {
    "orange": (
        (0.00, (0, 0, 0)),
        (0.12, (28, 3, 4)),
        (0.28, (94, 13, 10)),
        (0.48, (190, 45, 16)),
        (0.68, (245, 113, 18)),
        (0.84, (255, 205, 24)),
        (1.00, (255, 255, 255)),
    ),
    "gray": (
        (0.00, (0, 0, 0)),
        (1.00, (255, 255, 255)),
    ),
}


def read_theta_table(prejpg_path: Path, n_theta: int, offset: int = 68) -> np.ndarray:
    """Read n_theta little-endian float32 theta values from the pre-JPEG block."""
    data = prejpg_path.read_bytes()
    need = offset + n_theta * 4
    if len(data) < need:
        raise ValueError(f"{prejpg_path} is too short for {n_theta} theta floats at offset {offset}")
    theta = np.frombuffer(data[offset:need], dtype="<f4").astype(np.float64)

    # Basic sanity check. The table should be monotonic and roughly in radians.
    diffs = np.diff(theta)
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        raise ValueError(f"theta table in {prejpg_path} is not monotonic")
    if np.nanmax(np.abs(theta)) > math.pi * 1.25:
        raise ValueError(f"theta table in {prejpg_path} does not look like radians")
    return theta


def read_theta_csv(theta_csv_path: Path, n_theta: int) -> np.ndarray:
    """Read theta degrees from a saved live-viewer theta CSV and return radians."""
    values = []
    with theta_csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "theta_degrees" not in reader.fieldnames:
            raise ValueError(f"{theta_csv_path} must contain a theta_degrees column")
        for row in reader:
            values.append(float(row["theta_degrees"]))

    if len(values) != n_theta:
        raise ValueError(f"{theta_csv_path} has {len(values)} theta rows, but image has {n_theta} rows")

    theta = np.radians(np.asarray(values, dtype=np.float64))
    diffs = np.diff(theta)
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        raise ValueError(f"theta table in {theta_csv_path} is not monotonic")
    return theta


def bilinear_sample(img: np.ndarray, row_f: np.ndarray, col_f: np.ndarray, fill: int = 0) -> np.ndarray:
    """Sample a 2-D image at floating row/column coordinates using bilinear interpolation."""
    h, w = img.shape

    valid = (row_f >= 0) & (row_f <= h - 1) & (col_f >= 0) & (col_f <= w - 1)

    r0 = np.floor(np.clip(row_f, 0, h - 1)).astype(np.int32)
    c0 = np.floor(np.clip(col_f, 0, w - 1)).astype(np.int32)
    r1 = np.clip(r0 + 1, 0, h - 1)
    c1 = np.clip(c0 + 1, 0, w - 1)

    wr = np.clip(row_f - r0, 0.0, 1.0)
    wc = np.clip(col_f - c0, 0.0, 1.0)

    a = img[r0, c0].astype(np.float32)
    b = img[r0, c1].astype(np.float32)
    c = img[r1, c0].astype(np.float32)
    d = img[r1, c1].astype(np.float32)

    out = (1 - wr) * ((1 - wc) * a + wc * b) + wr * ((1 - wc) * c + wc * d)
    out = np.where(valid, out, fill)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_color_scheme(img: np.ndarray, color_scheme: str) -> Image.Image:
    """Map an 8-bit luminance image through a color ramp."""
    try:
        stops = COLOR_SCHEMES[color_scheme]
    except KeyError as exc:
        names = ", ".join(sorted(COLOR_SCHEMES))
        raise ValueError(f"unknown color scheme {color_scheme!r}; choose one of: {names}") from exc

    if color_scheme == "gray":
        return Image.fromarray(img, mode="L")

    x = img.astype(np.float32) / 255.0
    rgb = np.zeros((*img.shape, 3), dtype=np.float32)

    for (left_pos, left_color), (right_pos, right_color) in zip(stops, stops[1:]):
        mask = (x >= left_pos) & (x <= right_pos)
        if not np.any(mask):
            continue
        span = max(right_pos - left_pos, 1e-6)
        t = ((x[mask] - left_pos) / span)[:, None]
        left = np.array(left_color, dtype=np.float32)
        right = np.array(right_color, dtype=np.float32)
        rgb[mask] = left + (right - left) * t

    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def polar_image_to_xy(
    img: np.ndarray,
    theta: np.ndarray,
    out_path: Path,
    out_width: int = 900,
    out_height: int = 900,
    max_range: Optional[float] = None,
    range_offset_bins: float = 0.0,
    theta_offset_deg: float = 0.0,
    flip_theta: bool = False,
    flip_range: bool = False,
    forward_is_up: bool = True,
    color_scheme: str = "orange",
) -> Path:
    """Warp one polar theta/range image to X/Y raster.

    Coordinate convention:
      - r is range bin index by default, 0 at the transducer/origin.
      - theta is radians.
      - x = r * sin(theta)
      - y = r * cos(theta)

    Output image convention when forward_is_up=True:
      - +y / forward is toward the top of the image.
      - x positive is to the right.
    """
    n_theta, n_range = img.shape
    if flip_theta:
        theta = theta[::-1]
        img = img[::-1, :]
    if flip_range:
        img = img[:, ::-1]

    theta = theta + math.radians(theta_offset_deg)

    # Range units are arbitrary unless you know the real distance per bin.
    # Use bin index, optionally scaled to max_range.
    if max_range is None:
        r_max = float(n_range - 1 - range_offset_bins)
    else:
        r_max = float(max_range)
    r_min = 0.0

    theta_min = float(np.min(theta))
    theta_max = float(np.max(theta))

    # Build output X/Y coordinates. We include only the forward sector by default:
    # x spans the fan width at r_max; y spans 0..r_max.
    x_extent = r_max * max(abs(math.sin(theta_min)), abs(math.sin(theta_max)))
    y_extent = r_max

    x = np.linspace(-x_extent, x_extent, out_width)
    if forward_is_up:
        y = np.linspace(y_extent, 0.0, out_height)  # image row 0 = far/forward
    else:
        y = np.linspace(0.0, y_extent, out_height)
    X, Y = np.meshgrid(x, y)

    R = np.sqrt(X * X + Y * Y)
    TH = np.arctan2(X, Y)  # theta measured left/right from forward axis

    # Convert physical coordinates back into source image coordinates.
    # Row coordinate is obtained by interpolating theta -> row index.
    row_indices = np.arange(n_theta, dtype=np.float64)
    if theta[0] < theta[-1]:
        src_row = np.interp(TH, theta, row_indices, left=np.nan, right=np.nan)
    else:
        src_row = np.interp(TH, theta[::-1], row_indices[::-1], left=np.nan, right=np.nan)

    # Column coordinate is range-bin coordinate.
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

    warped = bilinear_sample(img, src_row, src_col, fill=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    apply_color_scheme(warped, color_scheme).save(out_path)
    return out_path


def polar_jpeg_to_xy(
    jpg_path: Path,
    prejpg_path: Path,
    out_path: Path,
    out_width: int = 900,
    out_height: int = 900,
    max_range: Optional[float] = None,
    range_offset_bins: float = 0.0,
    theta_offset_deg: float = 0.0,
    flip_theta: bool = False,
    flip_range: bool = False,
    forward_is_up: bool = True,
    color_scheme: str = "orange",
) -> Path:
    """Warp one extracted JPEG using theta from its .prejpg.bin metadata."""
    img = np.asarray(Image.open(jpg_path).convert("L"))
    theta = read_theta_table(prejpg_path, n_theta=img.shape[0])
    return polar_image_to_xy(
        img=img,
        theta=theta,
        out_path=out_path,
        out_width=out_width,
        out_height=out_height,
        max_range=max_range,
        range_offset_bins=range_offset_bins,
        theta_offset_deg=theta_offset_deg,
        flip_theta=flip_theta,
        flip_range=flip_range,
        forward_is_up=forward_is_up,
        color_scheme=color_scheme,
    )


def saved_live_frame_to_xy(
    image_path: Path,
    theta_csv_path: Path,
    out_path: Path,
    input_rotated_ccw: bool = False,
    out_width: int = 900,
    out_height: int = 900,
    max_range: Optional[float] = None,
    range_offset_bins: float = 0.0,
    theta_offset_deg: float = 0.0,
    flip_theta: bool = False,
    flip_range: bool = False,
    forward_is_up: bool = True,
    color_scheme: str = "orange",
) -> Path:
    """Warp a saved live-viewer raw PNG using theta from frame_XXXXXX_theta.csv."""
    img = np.asarray(Image.open(image_path).convert("L"))
    if input_rotated_ccw:
        img = np.rot90(img, k=-1)
    theta = read_theta_csv(theta_csv_path, n_theta=img.shape[0])
    return polar_image_to_xy(
        img=img,
        theta=theta,
        out_path=out_path,
        out_width=out_width,
        out_height=out_height,
        max_range=max_range,
        range_offset_bins=range_offset_bins,
        theta_offset_deg=theta_offset_deg,
        flip_theta=flip_theta,
        flip_range=flip_range,
        forward_is_up=forward_is_up,
        color_scheme=color_scheme,
    )


def default_pre_for_jpg(jpg: Path) -> Path:
    # frame_000123.jpg -> frame_000123.prejpg.bin
    return jpg.with_suffix(".prejpg.bin")


def iter_jobs(input_path: Path, pre: Optional[Path], out: Path) -> Iterable[Tuple[Path, Path, Path]]:
    if input_path.is_dir():
        out.mkdir(parents=True, exist_ok=True)
        for jpg in sorted(input_path.glob("*.jpg")):
            prejpg = default_pre_for_jpg(jpg)
            if prejpg.exists():
                yield jpg, prejpg, out / f"{jpg.stem}_xy.png"
    else:
        if pre is None:
            pre = default_pre_for_jpg(input_path)
        if out.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            out.mkdir(parents=True, exist_ok=True)
            out_file = out / f"{input_path.stem}_xy.png"
        else:
            out_file = out
        yield input_path, pre, out_file


def default_theta_csv_for_image(image_path: Path) -> Path:
    if image_path.name.endswith("_raw_rotated.png"):
        return image_path.with_name(image_path.name.replace("_raw_rotated.png", "_theta.csv"))
    return image_path.with_name(f"{image_path.stem}_theta.csv")


def iter_theta_csv_jobs(input_path: Path, theta_csv: Optional[Path], out: Path) -> Iterable[Tuple[Path, Path, Path]]:
    if input_path.is_dir():
        out.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(input_path.glob("*_raw_rotated.png")):
            theta_path = default_theta_csv_for_image(image_path)
            if theta_path.exists():
                yield image_path, theta_path, out / f"{image_path.stem}_xy.png"
    else:
        if theta_csv is None:
            theta_csv = default_theta_csv_for_image(input_path)
        if out.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            out.mkdir(parents=True, exist_ok=True)
            out_file = out / f"{input_path.stem}_xy.png"
        else:
            out_file = out
        yield input_path, theta_csv, out_file


def make_contact_sheet(paths: list[Path], out_path: Path, thumb_w: int = 240, cols: int = 4) -> None:
    if not paths:
        return
    thumbs = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        ratio = thumb_w / im.width
        im = im.resize((thumb_w, max(1, int(im.height * ratio))), Image.Resampling.LANCZOS)
        thumbs.append((p, im))
    rows = (len(thumbs) + cols - 1) // cols
    cell_h = max(im.height for _, im in thumbs) + 22
    sheet = Image.new("RGB", (cols * thumb_w, rows * cell_h), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for i, (p, im) in enumerate(thumbs):
        x = (i % cols) * thumb_w
        y = (i // cols) * cell_h
        sheet.paste(im, (x, y))
        draw.text((x + 3, y + im.height + 3), p.stem, fill=(220, 220, 220))
    sheet.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Warp extracted Garmin theta/range JPEGs into X/Y fan images.")
    ap.add_argument("input", type=Path, help="frame.jpg, frame_XXXXXX_raw_rotated.png, or a matching frame directory")
    ap.add_argument("--pre", type=Path, default=None, help="matching .prejpg.bin for single-image mode")
    ap.add_argument("--theta-csv", type=Path, default=None, help="matching frame_XXXXXX_theta.csv for saved live-viewer PNGs")
    ap.add_argument("--input-rotated-ccw", action="store_true", help="input image was saved rotated 90 degrees counter-clockwise")
    ap.add_argument("--out", type=Path, default=Path("xy_out"), help="output PNG path, or output directory for batch mode")
    ap.add_argument("--width", type=int, default=900, help="output image width")
    ap.add_argument("--height", type=int, default=900, help="output image height")
    ap.add_argument("--max-range", type=float, default=None, help="real-world max range; otherwise range units are source bins")
    ap.add_argument("--range-offset-bins", type=float, default=0.0, help="add this source-column offset to the range mapping")
    ap.add_argument("--theta-offset-deg", type=float, default=0.0, help="rotate theta table by this many degrees")
    ap.add_argument("--flip-theta", action="store_true", help="reverse theta rows if left/right is backwards")
    ap.add_argument("--flip-range", action="store_true", help="reverse range columns if near/far is backwards")
    ap.add_argument("--forward-down", action="store_true", help="put forward/far range at bottom instead of top")
    ap.add_argument(
        "--colorscheme",
        "--colourscheme",
        "--color-scheme",
        "--colour-scheme",
        default="orange",
        choices=sorted(COLOR_SCHEMES),
        help="output color ramp; default: orange",
    )
    ap.add_argument("--make-contact-sheet", action="store_true", help="batch mode: write xy_contact_sheet.jpg")
    args = ap.parse_args()

    written: list[Path] = []
    if args.theta_csv is not None or args.input_rotated_ccw:
        for image_path, theta_path, out_file in iter_theta_csv_jobs(args.input, args.theta_csv, args.out):
            print(f"warping {image_path.name} using {theta_path.name} -> {out_file}")
            written.append(
                saved_live_frame_to_xy(
                    image_path=image_path,
                    theta_csv_path=theta_path,
                    out_path=out_file,
                    input_rotated_ccw=args.input_rotated_ccw,
                    out_width=args.width,
                    out_height=args.height,
                    max_range=args.max_range,
                    range_offset_bins=args.range_offset_bins,
                    theta_offset_deg=args.theta_offset_deg,
                    flip_theta=args.flip_theta,
                    flip_range=args.flip_range,
                    forward_is_up=not args.forward_down,
                    color_scheme=args.colorscheme,
                )
            )
    else:
        for jpg, prejpg, out_file in iter_jobs(args.input, args.pre, args.out):
            print(f"warping {jpg.name} using {prejpg.name} -> {out_file}")
            written.append(
                polar_jpeg_to_xy(
                    jpg_path=jpg,
                    prejpg_path=prejpg,
                    out_path=out_file,
                    out_width=args.width,
                    out_height=args.height,
                    max_range=args.max_range,
                    range_offset_bins=args.range_offset_bins,
                    theta_offset_deg=args.theta_offset_deg,
                    flip_theta=args.flip_theta,
                    flip_range=args.flip_range,
                    forward_is_up=not args.forward_down,
                    color_scheme=args.colorscheme,
                )
            )

    print(f"wrote {len(written)} warped image(s)")
    if args.make_contact_sheet and written:
        sheet = args.out / "xy_contact_sheet.jpg" if args.out.suffix == "" else args.out.with_name("xy_contact_sheet.jpg")
        make_contact_sheet(written[:24], sheet)
        print(f"contact sheet: {sheet}")


if __name__ == "__main__":
    main()
