# Garmin LiveScope Tools

Python tools for decoding Garmin LiveScope/GLS 10-ish UDP image chunks, extracting the embedded JPEG frames, warping the polar sonar image into an X/Y fan view, and viewing or recording the stream live.

This repo currently has two main scripts:

- `garmin_livescope_live_viewer.py`: live packet sniffer, frame reassembler, OpenCV viewer, optional X/Y warp, optional frame/video recording.
- `warp_garmin_polar_to_xy.py`: converts extracted Garmin polar JPEG frames plus `.prejpg.bin` metadata into X/Y fan images.

The live viewer is split into helper modules:

- `garmin_packets.py`: packet header parsing, JPEG finding, and frame reassembly.
- `garmin_view.py`: raw rotation, raw recording frame prep, polar-to-X/Y warp, and color conversion.
- `garmin_state.py`: view state and motion/background subtraction.
- `garmin_echogram.py`: scrolling echogram image generation.
- `garmin_recording.py`: video writers, recording state, and autosave clips.
- `garmin_outputs.py`: save-folder prompts, safe names, unique paths, and frame dumps.
- `garmin_ui.py`: OpenCV window names, button rectangles, and button drawing.

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

Raw view, raw video recording, and saved raw frames are rotated 90 degrees counter-clockwise for display/use. The X/Y warp still uses the original decoded polar frame internally.

Warped X/Y fan view:

```bash
sudo python3 garmin_livescope_live_viewer.py --iface en9 --stream all --warp-xy
```

The viewer has a `WARP VIEW` / `RAW VIEW` button. Click it to switch between the raw Garmin JPEG and the warped X/Y fan view while the script is running. You can also press `w` while the OpenCV window is focused. `--warp-xy` only chooses the starting view.

The viewer also has a `MOTION ON` / `MOTION OFF` button. This removes the slowly changing background and shows only pixels that changed recently. In the main LiveScope view, motion color shows the direction of the brightness change: orange = brighter than the learned background, blue = darker than the learned background. You can also press `m` while the OpenCV window is focused.

Use the on-screen `-` and `+` buttons to lower or raise motion gain live. The gain value is shown between them; click the value to enter an exact number in the terminal. Keyboard shortcuts are `[`, `]`, and `g`.

Start with motion view already enabled:

```bash
sudo python3 garmin_livescope_live_viewer.py --iface en9 --stream all --motion
```

Useful motion tuning flags:

```bash
--motion-alpha 0.04
--motion-threshold 14
--motion-gain 4.0
```

Lower `--motion-threshold` shows weaker movement but may show more noise. Higher `--motion-gain` makes detected motion brighter. Lower `--motion-alpha` makes the background adapt more slowly.

The OpenCV window is created only after the first complete frame is decoded. If the script says it is listening but no window appears, it may not be receiving Garmin chunks on that interface yet.

The viewer also opens a second `Garmin LiveScope Echogram` window by default. It scrolls over time, adding one range-history column per decoded frame. Echogram background subtraction is always applied automatically. Brightness shows changing return strength, and color is automatically mapped from the beam/angle that produced the return: red = left, green = center, blue = right. The echogram window has its own `START REC` / `STOP REC` button that records echogram footage only. Hide it with:

```bash
sudo python3 garmin_livescope_live_viewer.py --iface en9 --stream all --no-echogram
```

Useful echogram options:

```bash
--echogram-width 700
--echogram-height 512
--echogram-mode max
--echogram-mode mean
--echogram-motion-threshold 14
--echogram-motion-gain 4.0
```

Stop the viewer with `Control-C` in the terminal, or press `q` while the OpenCV window is focused.

### Save Location

At startup, the live viewer asks where to save files, then asks for:

- raw MP4 filename
- echogram MP4 filename
- frames folder name

If a file or folder already exists, the script creates a new name like `test_002.mp4` or `frames_002` instead of overwriting old files.

Skip the prompt or set the folder from the command line:

```bash
--no-save-prompt
--save-root /path/to/all_sessions
--record-video raw_test.mp4
--record-echogram echo_test.mp4
--frames-dir frames_test
```

`--session-name trial_01` is also available as a quick prefix; it creates defaults like `trial_01_raw.mp4`, `trial_01_echogram.mp4`, and `trial_01_frames/`.

### Save Frames

Every decoded frame is saved by default to `frames/`:

```bash
frame_XXXXXX_raw_rotated.png  rotated raw view, lossless PNG
frame_XXXXXX_warp_metadata.txt  human-readable warp metadata, including estimated selected range in feet/meters
frame_XXXXXX_theta.csv        human-readable theta/angle table in degrees
```

Change or disable the frame dump:

```bash
--frames-dir frames
--no-save-frames
```

The older `--save` option is still available for display-oriented frame images:

```bash
sudo python3 garmin_livescope_live_viewer.py --iface en9 --stream all --save live_frames
```

### Record Video

The viewer always shows these controls:

- `START REC` / `STOP REC`: start or stop video recording.
- `WARP VIEW` / `RAW VIEW`: switch between raw and warped display.
- `MOTION ON` / `MOTION OFF`: show recent changes while suppressing static background.
- `-` / `+`: lower or raise motion gain.

By default, the main LiveScope recording is written to `livescope.mp4`. MP4 is still lossy, but the writer tries the best MP4 codecs OpenCV exposes first (`avc1`, `H264`) before falling back to `mp4v`. The saved PNG frames are the lossless copy. To choose a different raw output path:

```bash
sudo python3 garmin_livescope_live_viewer.py --iface en9 --stream all --record-video livescope.mp4
```

Click `START REC` in the main LiveScope window to start raw recording, click `STOP REC` to stop.

You can switch between raw and warped view while recording. The saved video always records the rotated raw decoded Garmin footage, without the warped display or on-screen buttons.

The echogram window records separately to `echogram.mp4` by default:

```bash
sudo python3 garmin_livescope_live_viewer.py --iface en9 --stream all --record-echogram echogram.mp4
```

Click `START REC` in the echogram window to record echogram footage only. The echogram recording includes the automatic background-subtracted view.

For automatic 10-minute autosave segments, use:

```bash
sudo python3 garmin_livescope_live_viewer.py --iface en9 --stream all --autosave-raw --autosave-echogram
```

Autosave uses `livescope.mp4`, `livescope_002.mp4`, etc. for raw footage and `echogram.mp4`, `echogram_002.mp4`, etc. for echogram footage. Change the paths or segment length with:

```bash
--record-video raw.mp4
--record-echogram echo.mp4
--autosave-minutes 10
```

Saved videos preserve real elapsed time by repeating the most recent frame when the Garmin stream arrives below the output `--video-fps`. This keeps a 10-minute autosave segment playing back as roughly 10 minutes instead of being compressed by low incoming FPS.

Start the viewer in warped view while still recording raw footage:

```bash
sudo python3 garmin_livescope_live_viewer.py --iface en9 --stream all --warp-xy --record-video livescope_xy.mp4
```

Set playback FPS in the output video:

```bash
sudo python3 garmin_livescope_live_viewer.py --iface en9 --stream all --warp-xy --record-video livescope_xy.mp4 --video-fps 30
```

`--video-fps` does not force the Garmin stream to produce that many frames. It controls how fast the saved video plays back.

Each start/stop creates a complete clip. The first recording uses the exact path you gave, such as `livescope_xy.mp4`; later recordings in the same run use names like `livescope_xy_002.mp4`, `livescope_xy_003.mp4`, and so on.

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
