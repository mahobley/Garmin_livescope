#!/usr/bin/env python3
"""
Garmin LiveScope-ish UDP chunk extractor.

What it does:
  1) Reads a .pcapng directly, without tshark/scapy.
  2) Finds UDP packets whose payload looks like the observed Garmin chunk header:
       uint32 magic      = 0x00000904
       uint32 len_after8 = udp_payload_len - 8
       uint32 stream_id  = often 0x00060044 for the image stream
       uint32 frame_id
       uint32 total_len
       uint32 chunk_offset
       uint32 chunk_len
       bytes  chunk_data
  3) Reassembles chunks into messages/frames.
  4) Searches each reassembled message for embedded JPEGs and writes them out.
  5) Creates contact sheets.
"""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

EPB = 0x00000006
SHB = 0x0A0D0D0A
IDB = 0x00000001

JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"


@dataclass
class UdpPacket:
    ts_us: int
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    payload: bytes


@dataclass
class ChunkHeader:
    magic: int
    len_after8: int
    stream_id: int
    frame_id: int
    total_len: int
    offset: int
    chunk_len: int


@dataclass
class ReassembledFrame:
    stream_id: int
    frame_id: int
    total_len: int
    data: bytes
    complete: bool
    received_bytes: int
    packet_count: int


def ip4(b: bytes) -> str:
    return ".".join(str(x) for x in b)


def iter_pcapng_packets(path: Path) -> Iterable[Tuple[int, bytes]]:
    """Yield (timestamp_us, captured_packet_bytes) for Enhanced Packet Blocks.

    Assumes little-endian pcapng, which is what Wireshark normally writes on x86.
    This is enough for the GS305E/Wireshark capture you sent.
    """
    data = path.read_bytes()
    off = 0
    endian = "<"
    while off + 12 <= len(data):
        block_type, block_len = struct.unpack_from(endian + "II", data, off)
        if block_len < 12 or off + block_len > len(data):
            break
        body = data[off + 8 : off + block_len - 4]
        if block_type == SHB:
            bom = body[:4]
            if bom == b"\x1a\x2b\x3c\x4d":
                endian = ">"
            else:
                endian = "<"
        elif block_type == EPB and len(body) >= 20:
            # interface_id, ts_high, ts_low, captured_len, original_len
            _ifid, ts_hi, ts_lo, cap_len, _orig_len = struct.unpack_from(endian + "IIIII", body, 0)
            pkt = body[20 : 20 + cap_len]
            ts_us = ((ts_hi << 32) | ts_lo)  # default timestamp unit is usually us
            yield ts_us, pkt
        off += block_len


def parse_udp_from_ethernet(ts_us: int, pkt: bytes) -> Optional[UdpPacket]:
    """Parse Ethernet II / optional VLAN / IPv4 / UDP and return payload."""
    if len(pkt) < 14:
        return None
    eth_type = struct.unpack_from("!H", pkt, 12)[0]
    l2 = 14
    if eth_type == 0x8100 and len(pkt) >= 18:  # VLAN tag
        eth_type = struct.unpack_from("!H", pkt, 16)[0]
        l2 = 18
    if eth_type != 0x0800 or len(pkt) < l2 + 20:
        return None
    ip0 = l2
    ver_ihl = pkt[ip0]
    if ver_ihl >> 4 != 4:
        return None
    ihl = (ver_ihl & 0x0F) * 4
    if len(pkt) < ip0 + ihl + 8:
        return None
    proto = pkt[ip0 + 9]
    if proto != 17:  # UDP
        return None
    total_len = struct.unpack_from("!H", pkt, ip0 + 2)[0]
    src = ip4(pkt[ip0 + 12 : ip0 + 16])
    dst = ip4(pkt[ip0 + 16 : ip0 + 20])
    u0 = ip0 + ihl
    sport, dport, ulen, _sum = struct.unpack_from("!HHHH", pkt, u0)
    payload = pkt[u0 + 8 : u0 + ulen]
    return UdpPacket(ts_us, src, dst, sport, dport, payload)


def parse_chunk_header(payload: bytes) -> Optional[ChunkHeader]:
    if len(payload) < 28:
        return None
    vals = struct.unpack_from("<IIIIIII", payload, 0)
    h = ChunkHeader(*vals)
    if h.magic != 0x00000904:
        return None
    if h.chunk_len < 0 or h.offset < 0 or h.total_len <= 0:
        return None
    if h.chunk_len > len(payload) - 28:
        return None
    # Observed: second word equals UDP payload length after the first 8 bytes.
    # Be permissive because final chunks can vary.
    if h.len_after8 not in (len(payload) - 8, len(payload) - 8 - 4, len(payload) - 8 + 4):
        pass
    if h.offset + h.chunk_len > h.total_len + 4096:
        return None
    return h


def reassemble_frames(udp_packets: Iterable[UdpPacket], stream_filter: Optional[int] = None) -> List[ReassembledFrame]:
    # key = (stream_id, frame_id, total_len)
    chunks: Dict[Tuple[int, int, int], Dict[int, bytes]] = {}
    counts: Dict[Tuple[int, int, int], int] = {}
    for p in udp_packets:
        h = parse_chunk_header(p.payload)
        if not h:
            continue
        if stream_filter is not None and h.stream_id != stream_filter:
            continue
        key = (h.stream_id, h.frame_id, h.total_len)
        chunks.setdefault(key, {})[h.offset] = p.payload[28 : 28 + h.chunk_len]
        counts[key] = counts.get(key, 0) + 1

    out: List[ReassembledFrame] = []
    for (stream_id, frame_id, total_len), parts in sorted(chunks.items(), key=lambda kv: kv[0][1]):
        buf = bytearray(total_len)
        mask = np.zeros(total_len, dtype=bool)
        for offset, data in parts.items():
            end = min(total_len, offset + len(data))
            if 0 <= offset < total_len:
                buf[offset:end] = data[: end - offset]
                mask[offset:end] = True
        out.append(
            ReassembledFrame(
                stream_id=stream_id,
                frame_id=frame_id,
                total_len=total_len,
                data=bytes(buf),
                complete=bool(mask.all()),
                received_bytes=int(mask.sum()),
                packet_count=counts[(stream_id, frame_id, total_len)],
            )
        )
    return out


def find_jpeg_span(data: bytes) -> Optional[Tuple[int, int]]:
    s = data.find(JPEG_SOI)
    if s < 0:
        return None
    e = data.find(JPEG_EOI, s + 2)
    if e < 0:
        return None
    return s, e + 2


def save_contact_sheet(paths: List[Path], out_path: Path, thumb_w: int = 256, cols: int = 4, max_images: int = 24) -> None:
    paths = paths[:max_images]
    if not paths:
        return
    thumbs = []
    for p in paths:
        im = Image.open(p).convert("L")
        ratio = thumb_w / im.width
        thumb_h = max(1, int(im.height * ratio))
        im = im.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        thumbs.append((p, im))
    rows = (len(thumbs) + cols - 1) // cols
    cell_h = max(t.height for _, t in thumbs) + 24
    sheet = Image.new("L", (cols * thumb_w, rows * cell_h), 0)
    draw = ImageDraw.Draw(sheet)
    for i, (p, im) in enumerate(thumbs):
        x = (i % cols) * thumb_w
        y = (i // cols) * cell_h
        sheet.paste(im, (x, y))
        draw.text((x + 4, y + im.height + 3), p.stem, fill=220)
    sheet.save(out_path)


def extract_jpegs(pcap: Path, outdir: Path, stream_filter: Optional[int] = 0x00060044, limit: Optional[int] = None) -> List[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    udp = []
    for ts, raw in iter_pcapng_packets(pcap):
        p = parse_udp_from_ethernet(ts, raw)
        if p is not None:
            udp.append(p)
    frames = reassemble_frames(udp, stream_filter=stream_filter)

    manifest = []
    jpg_paths: List[Path] = []
    for fr in frames:
        span = find_jpeg_span(fr.data)
        manifest.append(
            f"frame={fr.frame_id} stream=0x{fr.stream_id:08x} complete={fr.complete} "
            f"received={fr.received_bytes}/{fr.total_len} packets={fr.packet_count} jpeg_span={span}\n"
        )
        if not fr.complete or span is None:
            continue
        s, e = span
        jp = outdir / f"frame_{fr.frame_id:06d}.jpg"
        jp.write_bytes(fr.data[s:e])
        # Save pre/post bytes for reverse-engineering.
        (outdir / f"frame_{fr.frame_id:06d}.prejpg.bin").write_bytes(fr.data[:s])
        (outdir / f"frame_{fr.frame_id:06d}.postjpg.bin").write_bytes(fr.data[e:])
        jpg_paths.append(jp)
        if limit and len(jpg_paths) >= limit:
            break
    (outdir / "manifest.txt").write_text("".join(manifest))
    save_contact_sheet(jpg_paths, outdir / "contact_sheet.jpg")
    return jpg_paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pcapng", type=Path)
    ap.add_argument("--out", type=Path, default=Path("garmin_out"))
    ap.add_argument("--stream", default="0x00060044", help="stream id hex, or 'all'")
    ap.add_argument("--limit", type=int, default=None, help="max JPEGs to extract")
    args = ap.parse_args()

    stream = None if args.stream.lower() == "all" else int(args.stream, 0)
    jpgs = extract_jpegs(args.pcapng, args.out, stream_filter=stream, limit=args.limit)
    print(f"extracted {len(jpgs)} JPEGs to {args.out}")
    print(f"manifest: {args.out / 'manifest.txt'}")
    print(f"contact sheet: {args.out / 'contact_sheet.jpg'}")


if __name__ == "__main__":
    main()
