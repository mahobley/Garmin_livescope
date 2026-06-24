from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


BRIGHTER_MOTION_COLOR_BGR = np.array((0.0, 0.55, 1.0), dtype=np.float32)
DARKER_MOTION_COLOR_BGR = np.array((1.0, 0.25, 0.0), dtype=np.float32)


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

        motion = self.motion_difference(img)
        if motion is None:
            return np.zeros_like(img)

        return np.clip(np.abs(motion) * self.gain, 0, 255).astype(np.uint8)

    def apply_signed_color(self, img: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        motion = self.motion_difference(img)
        if motion is None:
            return np.zeros((*img.shape, 3), dtype=np.uint8)

        magnitude = np.clip(np.abs(motion) * self.gain, 0, 255).astype(np.uint8)
        display = np.zeros((*img.shape, 3), dtype=np.uint8)
        brighter = motion > 0
        darker = motion < 0

        # BGR colors: brighter-than-background is orange, darker is blue.
        display[brighter] = (
            magnitude[brighter].astype(np.float32)[:, None] * BRIGHTER_MOTION_COLOR_BGR
        ).astype(np.uint8)
        display[darker] = (
            magnitude[darker].astype(np.float32)[:, None] * DARKER_MOTION_COLOR_BGR
        ).astype(np.uint8)
        return display

    def motion_difference(self, img: np.ndarray) -> Optional[np.ndarray]:
        current = img.astype(np.float32)
        if self.background is None or self.background.shape != current.shape:
            self.background = current.copy()
            return None

        diff = current - self.background
        cv2.accumulateWeighted(current, self.background, self.alpha)

        magnitude = np.maximum(np.abs(diff) - float(self.threshold), 0.0)
        return np.sign(diff) * magnitude
