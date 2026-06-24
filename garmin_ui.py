from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from garmin_state import MotionState


WINDOW_NAME = "Garmin GLS 10 LiveScope"
ECHOGRAM_WINDOW_NAME = "Garmin LiveScope Echogram"
RECORD_BUTTON = (10, 42, 126, 78)
VIEW_BUTTON = (136, 42, 252, 78)
MOTION_BUTTON = (262, 42, 390, 78)
GAIN_DOWN_BUTTON = (10, 86, 48, 122)
GAIN_VALUE_RECT = (48, 86, 112, 122)
GAIN_UP_BUTTON = (112, 86, 150, 122)
MOTION_GAIN_STEP = 0.5


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
