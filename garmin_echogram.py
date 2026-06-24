from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from garmin_view import read_theta_table_from_prejpg


@dataclass
class EchogramState:
    width: int
    height: int
    mode: str = "max"
    image: Optional[np.ndarray] = None
    color_image: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.width = max(1, self.width)
        self.height = max(1, self.height)
        self.image = np.zeros((self.height, self.width), dtype=np.uint8)
        self.color_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def add_frame(self, polar_img: np.ndarray, prejpg: Optional[bytes] = None) -> np.ndarray:
        """Append one time column derived from the frame's range profile."""
        if self.image is None:
            self.image = np.zeros((self.height, self.width), dtype=np.uint8)
        if self.color_image is None:
            self.color_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if self.mode == "mean":
            profile = np.mean(polar_img, axis=0).astype(np.uint8)
            weights = polar_img.astype(np.float32)
            row_values = np.arange(polar_img.shape[0], dtype=np.float32)[:, None]
            weight_sum = np.maximum(np.sum(weights, axis=0), 1.0)
            angle_rows = np.sum(weights * row_values, axis=0) / weight_sum
        else:
            profile = np.max(polar_img, axis=0).astype(np.uint8)
            angle_rows = np.argmax(polar_img, axis=0).astype(np.float32)

        column = cv2.resize(profile[:, None], (1, self.height), interpolation=cv2.INTER_AREA)[:, 0]
        theta = self._theta_for_frame(polar_img, prejpg)
        angle_values = np.interp(angle_rows, np.arange(polar_img.shape[0], dtype=np.float32), theta)
        color_profile = self._angle_colors_bgr(angle_values)
        color_column = cv2.resize(color_profile[:, None, :], (1, self.height), interpolation=cv2.INTER_NEAREST)[:, 0, :]
        self.image[:, :-1] = self.image[:, 1:]
        self.image[:, -1] = column
        self.color_image[:, :-1, :] = self.color_image[:, 1:, :]
        self.color_image[:, -1, :] = color_column
        return self.image.copy()

    def render(self, intensity: Optional[np.ndarray] = None) -> np.ndarray:
        """Render the echogram with color carrying source-angle information."""
        if self.image is None:
            self.image = np.zeros((self.height, self.width), dtype=np.uint8)
        if self.color_image is None:
            self.color_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        value = self.image if intensity is None else intensity
        scale = value.astype(np.float32)[..., None] / 255.0
        return np.clip(self.color_image.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    @staticmethod
    def _theta_for_frame(polar_img: np.ndarray, prejpg: Optional[bytes]) -> np.ndarray:
        if prejpg is not None:
            try:
                return read_theta_table_from_prejpg(prejpg, n_theta=polar_img.shape[0])
            except ValueError:
                pass
        return np.linspace(-1.0, 1.0, polar_img.shape[0], dtype=np.float64)

    @staticmethod
    def _angle_colors_bgr(theta: np.ndarray) -> np.ndarray:
        max_abs = max(float(np.max(np.abs(theta))), 1e-6)
        t = np.clip(theta / max_abs, -1.0, 1.0)
        colors = np.zeros((theta.shape[0], 3), dtype=np.float32)

        left = t < 0
        colors[left, 1] = (1.0 + t[left]) * 255.0
        colors[left, 2] = 255.0

        right = ~left
        colors[right, 0] = t[right] * 255.0
        colors[right, 1] = (1.0 - t[right]) * 255.0

        return colors.astype(np.uint8)
