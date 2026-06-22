# Garmin LiveScope UDP/JPEG byte notes

Working notes for the two captures:

- `firsttest.pcapng`
- `secondtest.pcapng`

These notes describe what was observed in those captures only. Field names marked **inferred** are reverse-engineering hypotheses, not official Garmin documentation.

## High-level stream

Observed image stream:

```text
172.16.3.0:50223  →  239.254.2.16:50223
UDP multicast
stream_id = 0x00060044
```

Most packets in the image stream are UDP payload length `1428` bytes:

```text
28-byte chunk header + 1400-byte chunk payload = 1428-byte UDP payload
```

The large messages are split into chunks, then reassembled using `frame_id`, `total_len`, and `chunk_offset`.

---

## UDP chunk header

Every chunk starts with a 28-byte little-endian header.

```c
struct GarminUdpChunkHeader {
    uint32_t magic;        // 0x00000904
    uint32_t len_after8;   // UDP payload length minus 8
    uint32_t stream_id;    // 0x00060044 for observed image stream
    uint32_t frame_id;     // frame/message sequence number
    uint32_t total_len;    // total reassembled message length
    uint32_t offset;       // chunk offset within reassembled message
    uint32_t chunk_len;    // bytes of chunk_data in this packet
    uint8_t  chunk_data[];
};
```

### Chunk header table

| Offset | Size | Type | Example | Meaning |
|---:|---:|---|---|---|
| `0x00` | 4 | `uint32_le` | `04 09 00 00` → `0x00000904` | Magic / packet type marker |
| `0x04` | 4 | `uint32_le` | `8c 05 00 00` → `1420` | `len(udp_payload) - 8`; for 1428-byte UDP payloads this is 1420 |
| `0x08` | 4 | `uint32_le` | `44 00 06 00` → `0x00060044` | Stream ID; image stream in these captures |
| `0x0c` | 4 | `uint32_le` | varies | Frame/message ID |
| `0x10` | 4 | `uint32_le` | e.g. `42564`, `43276` | Total length of reassembled frame/message |
| `0x14` | 4 | `uint32_le` | `0`, `1400`, `2800`, ... | Offset of this chunk within reassembled frame |
| `0x18` | 4 | `uint32_le` | usually `1400` | Chunk payload length |
| `0x1c` | n | bytes | — | Chunk payload data |

### Reassembly key

Group chunks by:

```python
(stream_id, frame_id, total_len)
```

Then place each chunk at:

```python
message[offset : offset + chunk_len] = chunk_data
```

---

## Reassembled image message layout

For complete `0x00060044` frames in these captures, the reassembled message layout is:

```text
0x0000–0x0043   68-byte frame metadata header
0x0044–0x083f   511 × float32_le theta/angle table
0x0840–0x0e3c   zero padding, 1533 bytes
0x0e3d–...      embedded JPEG/JFIF image
...             optional post-JPEG footer/trailer
```

Fixed offsets observed:

| Region | Offset start | Length | Notes |
|---|---:|---:|---|
| Frame metadata header | `0x0000` / `0` | `68` bytes | Mixed integers/floats/config fields |
| Theta table | `0x0044` / `68` | `2044` bytes | `511 × float32_le` |
| Zero padding | `0x0840` / `2112` | `1533` bytes | All zero in both captures before normal JPEG start |
| JPEG start | `0x0e3d` / `3645` | variable | Starts with `FF D8 FF E0 ... JFIF` |
| Optional post-JPEG footer | after JPEG | `0` or `512` observed | Length appears to be stored in header field at `0x34` |

Note: one frame in `firsttest.pcapng` looked anomalous when searching for JPEG markers, but the normal image frames consistently used JPEG offset `3645`.

---

## 68-byte frame metadata header

The first 68 bytes of each reassembled message appear to be:

```c
struct GarminImageFrameHeader68 {
    uint32_t word00;             // 0x00, config/mode-ish
    float    value04;            // 0x04, dynamic per frame
    float    value08;            // 0x08, dynamic per frame
    uint32_t word0c;             // 0x0c, constant 0x69045951
    float    value10;            // 0x10, capture-dependent: 0.0 in secondtest, ~0.4799655 in firsttest
    uint32_t word14;             // 0x14, mode/config-ish
    uint32_t word18;             // 0x18, constant 0x02000000
    uint32_t table_count;        // 0x1c, 511
    float    value20;            // 0x20, range/scale-like; varies by capture/config
    uint32_t word24;             // 0x24, zero
    uint32_t jpeg_len;           // 0x28, length of embedded JPEG in bytes
    uint32_t word2c;             // 0x2c, zero
    uint32_t sentinel30;         // 0x30, 0xfe000000
    uint32_t post_jpeg_len;      // 0x34, bytes after JPEG; 0 or 512 observed
    uint32_t word38;             // 0x38, zero
    uint32_t word3c;             // 0x3c, 0x30440001 or 0x31440001 observed
    uint32_t word40;             // 0x40, 0x0003fe00
};
```

### Header byte table

| Offset | Size | Type guess | Observed values / behavior | Current interpretation |
|---:|---:|---|---|---|
| `0x00` | 4 | `uint32_le` | `0x644f0000` in `secondtest`; `0x64a30000`, `0x64500000`, `0x64a40000` in `firsttest` | Unknown config/mode/timestamp-ish word |
| `0x04` | 4 | `float32_le` | Dynamic every frame | Unknown dynamic value |
| `0x08` | 4 | `float32_le` | Dynamic every frame | Unknown dynamic value |
| `0x0c` | 4 | `uint32_le` | `0x69045951` | Constant/magic/config signature |
| `0x10` | 4 | `float32_le` or `uint32_le` | `0.0` in `secondtest`; `0.4799655378` in `firsttest` | Unknown; capture/config dependent |
| `0x14` | 4 | `uint32_le` | `0x01028787` in `secondtest`; `0x01048787` in `firsttest` | Mode/config word |
| `0x18` | 4 | `uint32_le` | `0x02000000` | Constant in both captures |
| `0x1c` | 4 | `uint32_le` | `511` / `0x000001ff` | Count of theta-table entries and image rows/height-like value |
| `0x20` | 4 | `float32_le` | `2.3847277164` in `secondtest`; multiple values in `firsttest`, including `3.2329814434` | Range/scale-like value; not decoded |
| `0x24` | 4 | `uint32_le` | `0` | Reserved/unused |
| `0x28` | 4 | `uint32_le` | Equals embedded JPEG byte length | **JPEG length** |
| `0x2c` | 4 | `uint32_le` | `0` | Reserved/unused |
| `0x30` | 4 | `uint32_le` | `0xfe000000` | Sentinel/config marker |
| `0x34` | 4 | `uint32_le` | `0` in `secondtest`; `512` in `firsttest` | **Post-JPEG footer length** |
| `0x38` | 4 | `uint32_le` | `0` | Reserved/unused |
| `0x3c` | 4 | `uint32_le` | `0x30440001` or `0x31440001` | Unknown config word |
| `0x40` | 4 | `uint32_le` | `0x0003fe00` | Constant/config word |

---

## Example 68-byte header: `secondtest.pcapng`

First complete frame inspected: `frame_id = 37988`

Raw 68 bytes:

```hex
00 00 4f 64 04 67 33 3d bc 3f a2 3f 51 59 04 69
00 00 00 00 87 87 02 01 00 00 00 02 ff 01 00 00
61 9f 18 40 00 00 00 00 07 98 00 00 00 00 00 00
00 00 00 fe 00 00 00 00 00 00 00 00 01 00 44 30
00 fe 03 00
```

Decoded as little-endian 32-bit words:

| Offset | Hex word | UInt32 | Float32 view |
|---:|---:|---:|---:|
| `0x00` | `0x644f0000` | `1682898944` | very large / not useful as float |
| `0x04` | `0x3d336704` | `1026778884` | `0.0437994152` |
| `0x08` | `0x3fa23fbc` | `1067597756` | `1.2675700188` |
| `0x0c` | `0x69045951` | `1761892689` | not useful as float |
| `0x10` | `0x00000000` | `0` | `0.0` |
| `0x14` | `0x01028787` | `16942983` | not useful as float |
| `0x18` | `0x02000000` | `33554432` | not useful as float |
| `0x1c` | `0x000001ff` | `511` | not useful as float |
| `0x20` | `0x40189f61` | `1075355489` | `2.3847277164` |
| `0x24` | `0x00000000` | `0` | `0.0` |
| `0x28` | `0x00009807` | `38919` | not useful as float |
| `0x2c` | `0x00000000` | `0` | `0.0` |
| `0x30` | `0xfe000000` | `4261412864` | sentinel-like |
| `0x34` | `0x00000000` | `0` | `0.0` |
| `0x38` | `0x00000000` | `0` | `0.0` |
| `0x3c` | `0x30440001` | `809762817` | not useful as float |
| `0x40` | `0x0003fe00` | `261632` | not useful as float |

For this frame:

```text
pre-JPEG length   = 3645 bytes
JPEG length       = 38919 bytes  // from header[0x28]
post-JPEG length  = 0 bytes      // from header[0x34]
total length      = 42564 bytes
```

Check:

```text
3645 + 38919 + 0 = 42564
```

---

## Example 68-byte header: `firsttest.pcapng`

Example frame inspected: `frame_id = 2974`

Raw 68 bytes:

```hex
00 00 a3 64 94 6d b5 be cb bb f4 3e 51 59 04 69
0b be f5 3e 87 87 04 01 00 00 00 02 ff 01 00 00
2b e9 4e 40 00 00 00 00 cf 98 00 00 00 00 00 00
00 00 00 fe 00 02 00 00 00 00 00 00 01 00 44 30
00 fe 03 00
```

Decoded highlights:

```text
header[0x10] = 0.4799655378
header[0x14] = 0x01048787
header[0x20] = 3.2329814434
header[0x28] = 39119  // JPEG length
header[0x34] = 512    // post-JPEG footer length
```

For this frame:

```text
pre-JPEG length   = 3645 bytes
JPEG length       = 39119 bytes
post-JPEG length  = 512 bytes
total length      = 43276 bytes
```

Check:

```text
3645 + 39119 + 512 = 43276
```

---

## Theta / angle table

The theta table starts immediately after the 68-byte header:

```text
theta_offset = 68
theta_count  = 511
theta_size   = 511 * 4 = 2044 bytes
theta_type   = float32 little-endian
```

Python:

```python
theta = np.frombuffer(frame_data[68:68 + 511*4], dtype="<f4")
```

### `secondtest.pcapng`

Observed theta values:

```text
theta[0]   = -1.3089969 rad  ≈ -75.0°
theta[510] = +1.3089969 rad  ≈ +75.0°
step       ≈ 0.005133271 rad ≈ 0.294115° per column
FOV        ≈ 150°
```

Formula approximation:

```python
theta[i] = -1.3089969 + i * ((2 * 1.3089969) / 510)
```

### `firsttest.pcapng`

Observed theta values:

```text
theta[0]   = -1.1780972 rad  ≈ -67.5°
theta[510] = +1.1780972 rad  ≈ +67.5°
step       ≈ 0.004619956 rad ≈ 0.264704° per column
FOV        ≈ 135°
```

So the theta table appears to encode the current horizontal/angular field of view.

---

## JPEG payload

The JPEG image starts at offset `3645` in normal complete frames:

```text
frame_data[3645:3645+4] = ff d8 ff e0
```

The bytes that follow identify a JFIF-style JPEG:

```text
ff d8 ff e0 00 10 4a 46 49 46 00 ...
```

Observed image dimensions:

```text
512 × 511
grayscale / luminance
```

Do not rely only on searching for `FF D9` to find the end of the JPEG. The 68-byte header already contains the JPEG byte length at offset `0x28`, which is a cleaner boundary:

```python
jpeg_start = 3645
jpeg_len   = struct.unpack_from("<I", frame_data, 0x28)[0]
jpeg_end   = jpeg_start + jpeg_len
jpeg_bytes = frame_data[jpeg_start:jpeg_end]
```

---

## Post-JPEG footer

The footer/trailer, when present, begins after the JPEG:

```python
post_start = jpeg_start + jpeg_len
post_len   = struct.unpack_from("<I", frame_data, 0x34)[0]
post_data  = frame_data[post_start:post_start + post_len]
```

Observed:

| Capture | `post_jpeg_len` | Actual bytes after JPEG |
|---|---:|---:|
| `secondtest.pcapng` | `0` | `0` |
| `firsttest.pcapng` | `512` | `512` |

The footer contents are not decoded yet. Since its length is exactly 512 in `firsttest`, it may be a per-row/per-column metadata block, palette/scaling block, or another fixed-size sidecar, but there is not enough evidence yet.

---

## Practical parser constants

```python
CHUNK_HEADER_LEN = 28
FRAME_HEADER_LEN = 68
THETA_OFFSET = 68
THETA_COUNT = 511
THETA_BYTES = THETA_COUNT * 4
JPEG_START = 3645

STREAM_IMAGE = 0x00060044
CHUNK_MAGIC = 0x00000904
```

Recommended parse flow:

```python
# after reassembling a frame:
header = frame_data[:68]

theta = np.frombuffer(frame_data[68:68 + 511*4], dtype="<f4")

jpeg_len = struct.unpack_from("<I", frame_data, 0x28)[0]
post_len = struct.unpack_from("<I", frame_data, 0x34)[0]

jpeg_start = 3645
jpeg_end = jpeg_start + jpeg_len
post_start = jpeg_end
post_end = post_start + post_len

jpeg_bytes = frame_data[jpeg_start:jpeg_end]
post_bytes = frame_data[post_start:post_end]

assert jpeg_start + jpeg_len + post_len == len(frame_data)
```

---

## Known unknowns / next reverse-engineering targets

1. **Header offsets `0x04` and `0x08`**  
   These are dynamic float-like values. They may be orientation, pan/tilt, stabilization, timestamp-derived values, range cursor, boat/ducer attitude, or sweep parameters.

2. **Header offset `0x10`**  
   `0.0` in `secondtest`, about `0.4799655` in `firsttest`. This likely changes with mode/config.

3. **Header offset `0x20`**  
   Looks like a useful float scale/range value. It is `2.3847` in `secondtest`; it varied in `firsttest`.

4. **Header offset `0x14`**  
   `0x01028787` vs `0x01048787`. This is likely a mode/config bitfield.

5. **Header offset `0x3c`**  
   `0x30440001` or `0x31440001`. Unknown config bit/flag.

6. **512-byte post-JPEG footer in `firsttest`**  
   This deserves comparison across frames. It may be a fixed sidecar table.

7. **Other UDP streams**  
   Smaller packets between chartplotter and GLS/LiveScope box may carry control/config commands that explain the header changes.
