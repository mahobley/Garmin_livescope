from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


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
