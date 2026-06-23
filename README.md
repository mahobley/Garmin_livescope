# Garmin LiveScope Tools

Python tools for decoding Garmin LiveScope/GLS 10-ish UDP image chunks, extracting the embedded JPEG frames, warping the polar sonar image into an X/Y fan view, and viewing or recording the stream live.

This repo currently has three main scripts:

- `garmin_livescope_live_viewer.py`: live packet sniffer, frame reassembler, OpenCV viewer, optional X/Y warp, optional frame/video recording.
- `garmin_pcap_decode_and_solve.py`: offline `.pcapng` decoder that extracts JPEG frames and metadata from a capture file.
- `warp_garmin_polar_to_xy.py`: converts extracted Garmin polar JPEG frames plus `.prejpg.bin` metadata into X/Y fan images.

## Install

Use Python 3.12+ if possible.

```bash
python3 -m pip install numpy pillow opencv-python scapy
```

On macOS/Linux, live sniffing usually requires `sudo` because Scapy needs raw packet capture access.

## Network Setup

For live viewing, your laptop must see the Ethernet traffic between the Garmin unit and chartplotter.

Example with a Netgear GS305E:

```text
Port 1: Garmin GLS 10 / LiveScope unit
Port 2: Garmin chartplotter
Port 5: laptop Ethernet adapter

Mirror source: port 1 and/or port 2
Mirror destination: port 5
Mirror direction: both ingress and egress, if the switch UI offers it
```

On the Mac used during development, the mirrored Ethernet adapter showed up as `en9`, while `en0` was the normal network/Wi-Fi interface.

List interfaces:

```bash
python3 -c "from scapy.all import get_if_list; print('\n'.join(get_if_list()))"
```

## Live Viewer

Raw decoded Garmin JPEG view:

```bash
sudo python3 garmin_livescope_live_viewer.py --iface en9 --stream all
```

Warped X/Y fan view:

```bash
sudo python3 garmin_livescope_live_viewer.py --iface en9 --stream all --warp-xy
```

The viewer has a `WARP VIEW` / `RAW VIEW` button. Click it to switch between the raw Garmin JPEG and the warped X/Y fan view while the script is running. You can also press `w` while the OpenCV window is focused. `--warp-xy` only chooses the starting view.

The OpenCV window is created only after the first complete frame is decoded. If the script says it is listening but no window appears, it may not be receiving Garmin chunks on that interface yet.

Stop the viewer with `Control-C` in the terminal, or press `q` while the OpenCV window is focused.

### Save Frames

Save decoded JPEG frames:

```bash
sudo python3 garmin_livescope_live_viewer.py --iface en9 --stream all --save live_frames
```

Save raw JPEGs and warped PNGs:

```bash
sudo python3 garmin_livescope_live_viewer.py --iface en9 --stream all --warp-xy --save live_frames
```

### Record Video

The viewer always shows both controls:

- `START REC` / `STOP REC`: start or stop video recording.
- `WARP VIEW` / `RAW VIEW`: switch between raw and warped display.

By default, recordings are written to `livescope.mp4`. To choose a different output path:

```bash
sudo python3 garmin_livescope_live_viewer.py --iface en9 --stream all --record-video livescope.mp4
```

Click `START REC` to start recording, click `STOP REC` to stop. You can also press `r` while the OpenCV window is focused.

You can switch between raw and warped view while recording. The current video clip keeps recording; if the display size changes, frames are resized to the clip size that was chosen when recording started.

Enable recording for the warped view:

```bash
sudo python3 garmin_livescope_live_viewer.py --iface en9 --stream all --warp-xy --record-video livescope_xy.mp4
```

Set playback FPS in the output video:

```bash
sudo python3 garmin_livescope_live_viewer.py --iface en9 --stream all --warp-xy --record-video livescope_xy.mp4 --video-fps 30
```

`--video-fps` does not force the Garmin stream to produce that many frames. It controls how fast the saved video plays back.

Each start/stop creates a complete clip. The first recording uses the exact path you gave, such as `livescope_xy.mp4`; later recordings in the same run use names like `livescope_xy_002.mp4`, `livescope_xy_003.mp4`, and so on.

## Decode a PCAPNG

Extract frames from a saved Wireshark/pcapng capture:

```bash
python3 garmin_pcap_decode_and_solve.py secondtest.pcapng --out garmin_out
```

Try all stream IDs if the default stream does not extract frames:

```bash
python3 garmin_pcap_decode_and_solve.py secondtest.pcapng --out garmin_out --stream all
```

Outputs include:

- `frame_XXXXXX.jpg`: extracted Garmin JPEG frame.
- `frame_XXXXXX.prejpg.bin`: metadata bytes before the JPEG, including the theta table used for warping.
- `frame_XXXXXX.postjpg.bin`: bytes after the JPEG, usually empty in the tested captures.
- `manifest.txt`: frame completeness and JPEG offset information.
- `contact_sheet.jpg`: quick visual preview of extracted frames.

## Warp Extracted Frames

Warp one extracted frame:

```bash
python3 warp_garmin_polar_to_xy.py garmin_out/frame_037988.jpg --pre garmin_out/frame_037988.prejpg.bin --out frame_037988_xy.png
```

Warp a whole output directory:

```bash
python3 warp_garmin_polar_to_xy.py garmin_out --out xy_out --make-contact-sheet
```

Useful tuning flags:

```bash
--width 900
--height 900
--colorscheme orange
--colorscheme gray
--flip-theta
--flip-range
--forward-down
--theta-offset-deg 5
--range-offset-bins 2
```

## Packet Format Assumption

The decoder looks for UDP payloads beginning with this little-endian chunk header:

```text
uint32 magic      = 0x00000904
uint32 len_after8 = udp_payload_len - 8
uint32 stream_id  = often 0x00060044 for the image stream
uint32 frame_id
uint32 total_len
uint32 chunk_offset
uint32 chunk_len
bytes  chunk_data
```

Chunks are reassembled by `(stream_id, frame_id, total_len)`. The complete frame is searched for JPEG start/end markers. Bytes before the JPEG are kept as `.prejpg.bin`; the warp tool expects a theta table there, starting at byte offset `68`.

## Troubleshooting

If `sudo` asks for a password and nothing appears while typing, that is normal. Type your Mac login password blindly and press Enter.

If you see `No /dev/bpf handle is available`, run the live viewer with `sudo`.

If no OpenCV window appears, check:

- You are using the mirrored Ethernet interface, likely `en9` on this Mac.
- The GS305E mirror destination is your laptop port.
- The mirror source includes the Garmin/chartplotter traffic, ideally both directions.
- The chartplotter is connected and actively requesting/displaying LiveScope data.
- Try `--stream all` in case the stream ID differs.

If the warped view is backwards or upside down, try `--flip-theta`, `--flip-range`, or `--forward-down`.

## Notes

This is a reverse-engineering/debugging tool based on observed captures. It is not an official Garmin API or supported interface.
