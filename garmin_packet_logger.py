from __future__ import annotations

import csv
import struct
import time
from pathlib import Path
from typing import Optional

import numpy as np
from scapy.all import IP, IPv6, UDP


TEMPERATURE_PACKET_PREFIX = b"\xd8\x07\x00\x00"
TEMPERATURE_PACKET_LENGTH = 18
TEMPERATURE_C_OFFSET = 10


class OtherPacketLogger:
    def __init__(self, outdir: Path, max_payload_bytes: int = 256):
        self.outdir = outdir
        self.max_payload_bytes = max_payload_bytes
        self.count = 0
        self.summary_path = outdir / "other_packets.csv"
        self.words_path = outdir / "other_packet_words.csv"
        self.summary_file = None
        self.words_file = None
        self.summary_writer = None
        self.words_writer = None

    def __enter__(self) -> "OtherPacketLogger":
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.summary_file = self.summary_path.open("w", newline="")
        self.words_file = self.words_path.open("w", newline="")
        self.summary_writer = csv.writer(self.summary_file)
        self.words_writer = csv.writer(self.words_file)
        self.summary_writer.writerow(
            [
                "packet_index",
                "unix_time",
                "src",
                "dst",
                "src_port",
                "dst_port",
                "payload_len",
                "first_bytes_hex",
            ]
        )
        self.words_writer.writerow(
            [
                "packet_index",
                "offset",
                "length",
                "hex",
                "uint32_le",
                "int32_le",
                "float32_le",
            ]
        )
        print(f"Logging non-image UDP packets to {self.outdir}")
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        if self.summary_file is not None:
            self.summary_file.close()
        if self.words_file is not None:
            self.words_file.close()
        print(f"Logged {self.count} non-image UDP packet(s) to {self.outdir}")

    def log(self, pkt, payload: bytes) -> None:
        if self.summary_writer is None or self.words_writer is None:
            return

        self.count += 1
        src, dst = packet_addresses(pkt)
        udp = pkt[UDP]
        self.summary_writer.writerow(
            [
                self.count,
                f"{time.time():.6f}",
                src,
                dst,
                int(udp.sport),
                int(udp.dport),
                len(payload),
                payload[: self.max_payload_bytes].hex(" "),
            ]
        )

        for offset in range(0, len(payload), 4):
            chunk = payload[offset:offset + 4]
            if len(chunk) == 4:
                uint32 = int.from_bytes(chunk, byteorder="little", signed=False)
                int32 = int.from_bytes(chunk, byteorder="little", signed=True)
                float32 = float(np.frombuffer(chunk, dtype="<f4", count=1)[0])
                self.words_writer.writerow(
                    [
                        self.count,
                        offset,
                        4,
                        chunk.hex(" "),
                        f"0x{uint32:08x}",
                        int32,
                        f"{float32:.9g}",
                    ]
                )
            else:
                self.words_writer.writerow([self.count, offset, len(chunk), chunk.hex(" "), "", "", ""])


def packet_addresses(pkt) -> tuple[str, str]:
    if IP in pkt:
        return str(pkt[IP].src), str(pkt[IP].dst)
    if IPv6 in pkt:
        return str(pkt[IPv6].src), str(pkt[IPv6].dst)
    return "", ""


def udp_payload(pkt) -> Optional[bytes]:
    if UDP not in pkt:
        return None
    return bytes(pkt[UDP].payload)


def extract_temperature_c(payload: bytes) -> Optional[float]:
    if len(payload) < TEMPERATURE_C_OFFSET + 4:
        return None
    if not payload.startswith(TEMPERATURE_PACKET_PREFIX):
        return None
    return struct.unpack_from("<f", payload, TEMPERATURE_C_OFFSET)[0]
