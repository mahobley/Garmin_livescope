#!/usr/bin/env python3
"""
Live Garmin GLS 10 / LiveScope JPEG viewer.

What it does:
  - Listens for Garmin UDP chunk packets on a network interface.
  - Reassembles chunks from stream 0x00060044 into frames.
  - Extracts the embedded JPEG.
  - Displays it live with OpenCV.
  - Optional: saves JPEG frames to disk.
  - Optional: records the displayed view to video.

Install:
  pip install scapy opencv-python numpy

Run:
  sudo python3 garmin_livescope_live_viewer.py --iface en0

Notes:
  - On macOS/Linux, packet sniffing usually requires sudo.
  - Use a managed switch with port mirroring between GLS 10 and chartplotter.
  - If the stream id differs, try --stream all first.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scapy.all import sniff

from garmin_echogram import EchogramState
from garmin_outputs import (
    append_timestamp_to_path,
    choose_save_root,
    prompt_output_path,
    safe_name,
    save_decoded_frame,
    under_save_root,
    unique_dir,
    unique_file,
)
from garmin_packet_logger import OtherPacketLogger, extract_temperature_c, udp_payload
from garmin_packets import GarminFrameAssembler, parse_chunk_header
from garmin_recording import RecordingState
from garmin_state import MotionState, ViewState
from garmin_ui import (
    ECHOGRAM_WINDOW_NAME,
    GAIN_DOWN_BUTTON,
    GAIN_UP_BUTTON,
    GAIN_VALUE_RECT,
    MOTION_BUTTON,
    MOTION_GAIN_STEP,
    RECORD_BUTTON,
    VIEW_BUTTON,
    WINDOW_NAME,
    draw_gain_buttons,
    draw_motion_button,
    draw_record_button,
    draw_view_button,
    prompt_motion_gain,
)
from garmin_view import COLOR_SCHEMES, colorize_for_cv2, polar_image_to_xy, prepare_raw_view_and_record_frame


def is_image_chunk(payload: bytes, stream_filter: Optional[int]) -> bool:
    header = parse_chunk_header(payload)
    if header is None:
        return False
    return stream_filter is None or header.stream_id == stream_filter


def decode_packet_frame(payload: bytes, assembler: GarminFrameAssembler) -> Optional[tuple[int, np.ndarray, bytes]]:
    result = assembler.add_payload(payload)
    if result is None:
        return None

    frame_id, jpg, prejpg = result
    arr = np.frombuffer(jpg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    return frame_id, img, prejpg


def update_echogram_window(
    img: np.ndarray,
    prejpg: bytes,
    echogram: EchogramState,
    echogram_motion: MotionState,
    echogram_recording: RecordingState,
) -> None:
    echogram_img = echogram.add_frame(img, prejpg)
    echogram_img = echogram_motion.apply(echogram_img)
    echogram_display = echogram.render(echogram_img)
    echogram_recording.sync((echogram_display.shape[1], echogram_display.shape[0]))
    echogram_recording.write(echogram_display)
    draw_record_button(
        echogram_display,
        recording_enabled=True,
        recording=echogram_recording.writer is not None,
    )
    cv2.imshow(ECHOGRAM_WINDOW_NAME, echogram_display)


def render_main_display(
    img: np.ndarray,
    prejpg: bytes,
    raw_view_img: np.ndarray,
    view: ViewState,
    motion: MotionState,
    args: argparse.Namespace,
    frame_id: int,
) -> Optional[np.ndarray]:
    if view.warp_enabled:
        try:
            warped = polar_image_to_xy(
                img,
                prejpg,
                out_width=args.width,
                out_height=args.height,
                max_range=args.max_range,
                range_offset_bins=args.range_offset_bins,
                theta_offset_deg=args.theta_offset_deg,
                flip_theta=args.flip_theta,
                flip_range=args.flip_range,
                forward_is_up=not args.forward_down,
            )
        except ValueError as exc:
            print(f"frame {frame_id}: cannot warp: {exc}")
            return None
        if motion.enabled:
            return motion.apply_signed_color(warped)
        return colorize_for_cv2(warped, args.colorscheme)

    return motion.apply_signed_color(raw_view_img)


def record_frame(recording: RecordingState, frame: np.ndarray) -> None:
    height, width = frame.shape[:2]
    recording.sync((width, height))
    recording.write(frame)


def draw_main_overlays(
    display: np.ndarray,
    frame_id: int,
    fps: float,
    recording: RecordingState,
    view: ViewState,
    motion: MotionState,
) -> None:
    cv2.putText(
        display,
        f"Frame {frame_id} | {fps:.1f} FPS",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )
    draw_record_button(display, recording_enabled=True, recording=recording.writer is not None)
    draw_view_button(display, warp_enabled=view.warp_enabled)
    draw_motion_button(display, motion_enabled=motion.enabled)
    draw_gain_buttons(display, gain=motion.gain)


def handle_keyboard(key: int, view: ViewState, motion: MotionState) -> None:
    if key == ord("w"):
        view.toggle()
    if key == ord("m"):
        motion.toggle()
    if key == ord("["):
        motion.adjust_gain(-MOTION_GAIN_STEP)
    if key == ord("]"):
        motion.adjust_gain(MOTION_GAIN_STEP)
    if key == ord("g"):
        prompt_motion_gain(motion)
    if key == ord("q"):
        raise KeyboardInterrupt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iface", required=True, help="Network interface, e.g. en0, eth0, Ethernet")
    parser.add_argument("--stream", default="0x00060044", help="Stream ID hex, or 'all'")
    parser.add_argument("--save", type=Path, default=None, help="Optional folder to save display-oriented frame images")
    parser.add_argument("--frames-dir", type=Path, default=Path("frames"), help="Folder to save every rotated raw frame as PNG; default: frames")
    parser.add_argument("--no-save-frames", action="store_true", help="Disable saving every decoded frame to --frames-dir")
    parser.add_argument("--record-video", type=Path, default=Path("livescope.mp4"), help="Output path for the START REC button; default: livescope.mp4")
    parser.add_argument("--record-echogram", type=Path, default=Path("echogram.mp4"), help="Output path for the echogram START REC button; default: echogram.mp4")
    parser.add_argument("--video-fps", type=float, default=20.0, help="FPS to write into recorded video; default: 20")
    parser.add_argument("--save-root", type=Path, default=None, help="Root folder for relative output paths; prompts by default")
    parser.add_argument("--session-name", default=None, help="Optional prefix for default output names")
    parser.add_argument("--no-save-prompt", action="store_true", help="Do not ask for an output folder at startup")
    parser.add_argument("--autosave-raw", action="store_true", help="Automatically record raw footage in rolling segments")
    parser.add_argument("--autosave-echogram", action="store_true", help="Automatically record echogram footage in rolling segments")
    parser.add_argument("--autosave-minutes", type=float, default=10.0, help="Autosave segment length in minutes; default: 10")
    parser.add_argument("--log-other-packets", action="store_true", help="Log non-image UDP packets for reverse engineering")
    parser.add_argument("--other-packets-dir", type=Path, default=Path("other_packets"), help="Folder for --log-other-packets CSV output")
    parser.add_argument("--udp-port", type=int, default=None, help="Optional UDP port filter")
    parser.add_argument("--warp-xy", action="store_true", help="Start in warped theta/range X/Y fan view")
    parser.add_argument("--width", type=int, default=900, help="warped output width when --warp-xy is enabled")
    parser.add_argument("--height", type=int, default=900, help="warped output height when --warp-xy is enabled")
    parser.add_argument("--max-range", type=float, default=None, help="real-world max range; otherwise range units are source bins")
    parser.add_argument("--range-offset-bins", type=float, default=0.0, help="add this source-column offset to the range mapping")
    parser.add_argument("--theta-offset-deg", type=float, default=0.0, help="rotate theta table by this many degrees")
    parser.add_argument("--flip-theta", action="store_true", help="reverse theta rows if left/right is backwards")
    parser.add_argument("--flip-range", action="store_true", help="reverse range columns if near/far is backwards")
    parser.add_argument("--forward-down", action="store_true", help="put forward/far range at bottom instead of top")
    parser.add_argument("--motion", action="store_true", help="Start with background-subtraction motion view enabled")
    parser.add_argument("--motion-alpha", type=float, default=0.04, help="Background learning rate for motion view; default: 0.04")
    parser.add_argument("--motion-threshold", type=int, default=14, help="Minimum brightness change shown in motion view; default: 14")
    parser.add_argument("--motion-gain", type=float, default=4.0, help="Brightness gain applied to motion differences; default: 4.0")
    parser.add_argument("--no-echogram", action="store_true", help="Disable the second scrolling echogram window")
    parser.add_argument("--echogram-width", type=int, default=700, help="Scrolling echogram window width; default: 700")
    parser.add_argument("--echogram-height", type=int, default=512, help="Scrolling echogram window height; default: 512")
    parser.add_argument("--echogram-mode", choices=("max", "mean"), default="max", help="Range profile reducer for echogram columns; default: max")
    parser.add_argument("--echogram-motion-alpha", type=float, default=0.04, help="Echogram background learning rate; default: 0.04")
    parser.add_argument("--echogram-motion-threshold", type=int, default=14, help="Minimum echogram brightness change shown; default: 14")
    parser.add_argument("--echogram-motion-gain", type=float, default=4.0, help="Brightness gain for echogram differences; default: 4.0")
    parser.add_argument(
        "--colorscheme",
        "--colourscheme",
        "--color-scheme",
        "--colour-scheme",
        default="orange",
        choices=sorted(COLOR_SCHEMES),
        help="warped display color ramp; default: orange",
    )
    args = parser.parse_args()
    run_timestamp = time.strftime("%Y-%m-%d_%H%M%S")

    stream_filter = None if args.stream.lower() == "all" else int(args.stream, 0)
    assembler = GarminFrameAssembler(stream_filter=stream_filter)

    save_root = args.save_root
    if save_root is None and not args.no_save_prompt:
        save_root = choose_save_root()
    if save_root is not None:
        save_root.mkdir(parents=True, exist_ok=True)
        if args.session_name:
            prefix = safe_name(args.session_name)
            args.record_video = Path(f"{prefix}_raw_{run_timestamp}{args.record_video.suffix or '.mp4'}")
            args.record_echogram = Path(f"{prefix}_echogram_{run_timestamp}{args.record_echogram.suffix or '.mp4'}")
            args.frames_dir = Path(f"{prefix}_frames_{run_timestamp}")
            args.other_packets_dir = Path(f"{prefix}_other_packets_{run_timestamp}")
        elif not args.no_save_prompt:
            args.record_video = prompt_output_path("Raw MP4 filename", args.record_video)
            args.record_echogram = prompt_output_path("Echogram MP4 filename", args.record_echogram)
            args.frames_dir = prompt_output_path("Frames folder name", args.frames_dir, is_dir=True)
            args.record_video = append_timestamp_to_path(args.record_video, run_timestamp)
            args.record_echogram = append_timestamp_to_path(args.record_echogram, run_timestamp)
            args.frames_dir = append_timestamp_to_path(args.frames_dir, run_timestamp)
            args.other_packets_dir = append_timestamp_to_path(args.other_packets_dir, run_timestamp)
        else:
            args.record_video = append_timestamp_to_path(args.record_video, run_timestamp)
            args.record_echogram = append_timestamp_to_path(args.record_echogram, run_timestamp)
            args.frames_dir = append_timestamp_to_path(args.frames_dir, run_timestamp)
            args.other_packets_dir = append_timestamp_to_path(args.other_packets_dir, run_timestamp)

        args.save = under_save_root(args.save, save_root)
        args.frames_dir = unique_dir(under_save_root(args.frames_dir, save_root))
        args.other_packets_dir = unique_dir(under_save_root(args.other_packets_dir, save_root))
        args.record_video = unique_file(under_save_root(args.record_video, save_root))
        args.record_echogram = unique_file(under_save_root(args.record_echogram, save_root))
        print(f"Saving outputs under: {save_root}")
        print(f"Raw video: {args.record_video}")
        print(f"Echogram video: {args.record_echogram}")
        print(f"Frames folder: {args.frames_dir}")
        if args.log_other_packets:
            print(f"Other-packet logs: {args.other_packets_dir}")
    else:
        args.record_video = append_timestamp_to_path(args.record_video, run_timestamp)
        args.record_echogram = append_timestamp_to_path(args.record_echogram, run_timestamp)
        args.frames_dir = append_timestamp_to_path(args.frames_dir, run_timestamp)
        args.other_packets_dir = append_timestamp_to_path(args.other_packets_dir, run_timestamp)

    save_all_frames = not args.no_save_frames
    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
    if save_all_frames:
        args.frames_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    t0 = time.time()
    latest_temperature_c = None
    printed_temperature_detected = False
    autosave_seconds = max(1.0, args.autosave_minutes * 60.0)
    recording = RecordingState(
        args.record_video,
        args.video_fps,
        label="raw footage",
        auto_segment_seconds=autosave_seconds if args.autosave_raw else None,
    )
    echogram_recording = RecordingState(
        args.record_echogram,
        args.video_fps,
        label="echogram footage",
        auto_segment_seconds=autosave_seconds if args.autosave_echogram else None,
    )
    view = ViewState(warp_enabled=args.warp_xy)
    echogram = None if args.no_echogram else EchogramState(
        width=args.echogram_width,
        height=args.echogram_height,
        mode=args.echogram_mode,
    )
    motion = MotionState(
        enabled=args.motion,
        alpha=float(np.clip(args.motion_alpha, 0.0, 1.0)),
        threshold=max(0, args.motion_threshold),
        gain=max(0.0, args.motion_gain),
    )
    echogram_motion = MotionState(
        enabled=True,
        alpha=float(np.clip(args.echogram_motion_alpha, 0.0, 1.0)),
        threshold=max(0, args.echogram_motion_threshold),
        gain=max(0.0, args.echogram_motion_gain),
    )

    print("Listening for Garmin LiveScope packets...")
    print("Press q in the OpenCV window to quit.")
    print("Click WARP VIEW/RAW VIEW in the OpenCV window, or press w, to switch display modes.")
    print("Click MOTION ON/OFF in the OpenCV window, or press m, to toggle background subtraction.")
    print("Click -/+/gain value in the OpenCV window, or press [, ], and g, to adjust motion gain.")
    print("Click START REC in the main OpenCV window to start/stop raw recording.")
    if args.autosave_raw:
        print(f"Autosaving raw footage every {args.autosave_minutes:g} minute(s) to {args.record_video}")
    if args.autosave_echogram:
        print(f"Autosaving echogram footage every {args.autosave_minutes:g} minute(s) to {args.record_echogram}")
    if echogram is not None:
        print("Showing scrolling echogram in a second OpenCV window. Its START REC button records echogram footage.")
    if args.log_other_packets:
        print("Logging non-image UDP packets for reverse engineering.")

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        x1, y1, x2, y2 = VIEW_BUTTON
        if x1 <= x <= x2 and y1 <= y <= y2:
            view.toggle()
            return

        x1, y1, x2, y2 = MOTION_BUTTON
        if x1 <= x <= x2 and y1 <= y <= y2:
            motion.toggle()
            return

        x1, y1, x2, y2 = GAIN_DOWN_BUTTON
        if x1 <= x <= x2 and y1 <= y <= y2:
            motion.adjust_gain(-MOTION_GAIN_STEP)
            return

        x1, y1, x2, y2 = GAIN_VALUE_RECT
        if x1 <= x <= x2 and y1 <= y <= y2:
            prompt_motion_gain(motion)
            return

        x1, y1, x2, y2 = GAIN_UP_BUTTON
        if x1 <= x <= x2 and y1 <= y <= y2:
            motion.adjust_gain(MOTION_GAIN_STEP)
            return

        if recording is not None:
            x1, y1, x2, y2 = RECORD_BUTTON
            if x1 <= x <= x2 and y1 <= y <= y2:
                recording.toggle_requested()

    def on_echogram_mouse(event, x, y, _flags, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        x1, y1, x2, y2 = RECORD_BUTTON
        if x1 <= x <= x2 and y1 <= y <= y2:
            echogram_recording.toggle_requested()
            return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    cv2.moveWindow(WINDOW_NAME, 20, 40)
    if echogram is not None:
        cv2.namedWindow(ECHOGRAM_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(ECHOGRAM_WINDOW_NAME, on_echogram_mouse)
        cv2.resizeWindow(ECHOGRAM_WINDOW_NAME, echogram.width, echogram.height)
        cv2.moveWindow(ECHOGRAM_WINDOW_NAME, 560, 40)

    bpf = "udp"
    if args.udp_port is not None:
        bpf = f"udp port {args.udp_port}"

    def handle_packet(pkt) -> None:
        nonlocal frame_count, t0, latest_temperature_c, printed_temperature_detected

        payload = udp_payload(pkt)
        if payload is None:
            return

        temperature_c = extract_temperature_c(payload)
        if temperature_c is not None:
            latest_temperature_c = temperature_c
            if not printed_temperature_detected:
                temperature_f = temperature_c * 9.0 / 5.0 + 32.0
                print(f"Temperature detected: {temperature_c:.2f} C / {temperature_f:.2f} F")
                printed_temperature_detected = True

        if packet_logger is not None and not is_image_chunk(payload, stream_filter):
            packet_logger.log(pkt, payload)
            return

        result = decode_packet_frame(payload, assembler)
        if result is None:
            return

        frame_id, img, prejpg = result

        if echogram is not None:
            update_echogram_window(img, prejpg, echogram, echogram_motion, echogram_recording)

        raw_view_img, raw_record_frame = prepare_raw_view_and_record_frame(img)

        if save_all_frames:
            save_decoded_frame(
                args.frames_dir,
                frame_id,
                raw_view_img,
                prejpg,
                img.shape,
                capture_time=time.time(),
                temperature_c=latest_temperature_c,
            )

        frame_count += 1
        elapsed = max(0.001, time.time() - t0)
        fps = frame_count / elapsed

        display = render_main_display(img, prejpg, raw_view_img, view, motion, args, frame_id)
        if display is None:
            return

        if recording:
            record_frame(recording, raw_record_frame)

        if recording:
            draw_main_overlays(display, frame_id, fps, recording, view, motion)

        cv2.imshow(WINDOW_NAME, display)

        if args.save:
            out = args.save / f"frame_{frame_id:06d}.jpg"
            cv2.imwrite(str(out), raw_view_img)
            if view.warp_enabled:
                warped_out = args.save / f"frame_{frame_id:06d}_xy.png"
                cv2.imwrite(str(warped_out), display)

        key = cv2.waitKey(1) & 0xFF
        handle_keyboard(key, view, motion)

    packet_logger = OtherPacketLogger(args.other_packets_dir) if args.log_other_packets else None
    try:
        if packet_logger is None:
            sniff(iface=args.iface, filter=bpf, prn=handle_packet, store=False)
        else:
            with packet_logger:
                sniff(iface=args.iface, filter=bpf, prn=handle_packet, store=False)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if recording is not None:
            recording.stop()
        if echogram is not None:
            echogram_recording.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
