from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"


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
