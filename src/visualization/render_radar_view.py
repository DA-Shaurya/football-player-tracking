import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict, deque
import cv2
import numpy as np
import pandas as pd

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analytics.pitch_model import FootballPitch
from src.visualization.render_tracking import get_color

def render_radar_video(
    video_path,
    trajectory_csv,
    output_path,
    layout="side_by_side",  # "side_by_side" or "pip" (picture-in-picture)
    radar_width=640,
    radar_height=415,
    trail_len=20
):
    video_path = Path(video_path)
    trajectory_csv = Path(trajectory_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RENDERING 2D TACTICAL BIRD'S-EYE RADAR VIDEO")
    print(f"Source Video: {video_path}")
    print(f"Trajectory CSV: {trajectory_csv}")
    print(f"Output Video: {output_path}")
    print(f"Layout Mode: {layout}")
    print("=" * 80)

    # 1. Load Trajectory Data
    df = pd.read_csv(trajectory_csv)
    frames_dict = defaultdict(list)
    for _, row in df.iterrows():
        frames_dict[int(row["frame"])].append({
            "id": int(row["track_id"]),
            "x_pixel": float(row["x_pixel"]),
            "y_pixel": float(row["y_pixel"]),
            "x_pitch": float(row["x_pitch_smooth"] if "x_pitch_smooth" in row else row["x_pitch"]),
            "y_pitch": float(row["y_pitch_smooth"] if "y_pitch_smooth" in row else row["y_pitch"]),
            "is_in_pitch": bool(row["is_in_pitch"]),
            "conf": float(row["confidence"])
        })

    # 2. Open Video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps): fps = 25.0
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Base pitch canvas
    base_pitch, scale_x, scale_y, pad = FootballPitch.draw_2d_pitch(
        canvas_width=radar_width,
        canvas_height=radar_height,
        padding=25,
        bg_color=(28, 90, 38),
        line_color=(255, 255, 255)
    )
    pitch_h, pitch_w = base_pitch.shape[:2]

    # Target output dimensions
    if layout == "side_by_side":
        # Scale video height to match pitch or match height
        target_h = 720
        # Resize video to 1280x720
        vid_disp_w = 1280
        vid_disp_h = 720
        # Scale pitch to same height
        pitch_scale = target_h / pitch_h
        disp_pitch_w = int(pitch_w * pitch_scale)
        disp_pitch_h = target_h
        out_w = vid_disp_w + disp_pitch_w
        out_h = target_h
    else:  # PIP
        out_w = vid_w
        out_h = vid_h
        pip_w = int(vid_w * 0.35)
        pip_h = int(pitch_h * (pip_w / pitch_w))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))
    if not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")

    # Track pitch trajectory trails
    pitch_trajectories = defaultdict(lambda: deque(maxlen=trail_len))
    t0 = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        active_dets = frames_dict.get(frame_idx, [])
        active_ids = set()

        # Update active pitch points
        for det in active_dets:
            tid = det["id"]
            active_ids.add(tid)
            # Convert pitch meter to pitch canvas pixel
            cx = int(pad + det["x_pitch"] * scale_x)
            cy = int(pad + det["y_pitch"] * scale_y)
            # Clamp inside pitch canvas display
            cx = max(2, min(pitch_w - 3, cx))
            cy = max(2, min(pitch_h - 3, cy))
            pitch_trajectories[tid].append((cx, cy))

        # Copy fresh pitch background
        pitch_frame = base_pitch.copy()

        # Draw metric trajectory trails on pitch
        for tid, pts in list(pitch_trajectories.items()):
            if tid not in active_ids and len(pts) > 0:
                pts.popleft()
                if len(pts) == 0:
                    del pitch_trajectories[tid]
                    continue
            color = get_color(tid)
            pts_list = list(pts)
            for i in range(1, len(pts_list)):
                alpha = i / len(pts_list)
                th = max(1, int(round(1 + 2 * alpha)))
                cv2.line(pitch_frame, pts_list[i-1], pts_list[i], color, th)

        # Draw player dots on 2D tactical board
        for det in active_dets:
            tid = det["id"]
            color = get_color(tid)
            cx = int(pad + det["x_pitch"] * scale_x)
            cy = int(pad + det["y_pitch"] * scale_y)

            # In-pitch vs Out-of-pitch styling
            if det["is_in_pitch"]:
                # Solid circular badge
                cv2.circle(pitch_frame, (cx, cy), 8, color, -1)
                cv2.circle(pitch_frame, (cx, cy), 9, (255, 255, 255), 1)
                # Player ID inside/above
                cv2.putText(pitch_frame, str(tid), (cx - 6, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                # Dimmed badge for sideline/touchline players
                cv2.circle(pitch_frame, (cx, cy), 5, (180, 180, 180), -1)

        # Header on pitch canvas
        cv2.putText(pitch_frame, "2D TACTICAL RADAR (105m x 68m)", (pad, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(pitch_frame, f"Active Players: {len(active_dets)}", (pitch_w - 160, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Composite video output
        if layout == "side_by_side":
            vid_resized = cv2.resize(frame, (vid_disp_w, vid_disp_h))
            pitch_resized = cv2.resize(pitch_frame, (disp_pitch_w, disp_pitch_h))
            combined = np.hstack([vid_resized, pitch_resized])
            out.write(combined)
        else:  # Picture in picture
            pip_resized = cv2.resize(pitch_frame, (pip_w, pip_h))
            # Place in bottom-right corner with 15px margin
            y_start = vid_h - pip_h - 15
            x_start = vid_w - pip_w - 15
            # Draw white border around PIP
            cv2.rectangle(frame, (x_start - 2, y_start - 2), (x_start + pip_w + 2, y_start + pip_h + 2), (255, 255, 255), 2)
            frame[y_start:y_start+pip_h, x_start:x_start+pip_w] = pip_resized
            out.write(frame)

        if frame_idx % 500 == 0 or frame_idx == total_frames:
            elapsed = time.time() - t0
            cur_fps = frame_idx / elapsed if elapsed > 0 else 0
            print(f"  Rendered {frame_idx}/{total_frames} frames ({cur_fps:.1f} FPS)...")

    cap.release()
    out.release()
    total_time = time.time() - t0
    print(f"\n2D Tactical Radar Video saved to {output_path} ({total_time:.1f}s, {frame_idx/total_time:.1f} FPS)")

def main():
    parser = argparse.ArgumentParser(description="Render 2D tactical radar video from metric pitch trajectories.")
    parser.add_argument("--video", required=True, help="Input video file path")
    parser.add_argument("--trajectories", required=True, help="Trajectory CSV with x_pitch, y_pitch")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--layout", default="side_by_side", choices=["side_by_side", "pip"], help="Display layout")
    args = parser.parse_args()

    render_radar_video(args.video, args.trajectories, args.output, layout=args.layout)

if __name__ == "__main__":
    main()
