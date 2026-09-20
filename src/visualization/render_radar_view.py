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
from src.analytics.dynamic_pitch_mapper import DynamicPitchMapper
from src.visualization.render_tracking import get_color

def render_radar_video(
    video_path,
    trajectory_csv,
    output_path,
    layout="side_by_side",  # "side_by_side" or "pip"
    radar_width=640,
    radar_height=415,
    trail_len=20,
    use_dynamic_mapper=True,
    min_tactical_players=4
):
    video_path = Path(video_path)
    trajectory_csv = Path(trajectory_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RENDERING DYNAMIC 2D TACTICAL BIRD'S-EYE RADAR VIDEO")
    print(f"Source Video: {video_path}")
    print(f"Tracking/Trajectory CSV: {trajectory_csv}")
    print(f"Output Video: {output_path}")
    print(f"Layout Mode: {layout} | Dynamic Compensation: {use_dynamic_mapper}")
    print("=" * 80)

    # 1. Load Detections Data
    df = pd.read_csv(trajectory_csv)
    frames_dict = defaultdict(list)
    for _, row in df.iterrows():
        # Handle both raw tracking CSV (x1, y1, x2, y2) and trajectory CSV (x_pixel, y_pixel)
        if "x_pixel" in row:
            x_mid = float(row["x_pixel"])
            y_mid = float(row["y_pixel"])
        else:
            x_mid = float((row["x1"] + row["x2"]) / 2.0)
            y_mid = float(row["y2"])

        x_pitch = float(row["x_pitch"]) if "x_pitch" in row else None
        y_pitch = float(row["y_pitch"]) if "y_pitch" in row else None

        frames_dict[int(row["frame"])].append({
            "id": int(row["track_id"]),
            "x_pixel": x_mid,
            "y_pixel": y_mid,
            "x_pitch_static": x_pitch,
            "y_pitch_static": y_pitch,
            "conf": float(row["confidence"])
        })

    # 2. Open Video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps): fps = 29.97
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
        target_h = 720
        vid_disp_w = 1280
        vid_disp_h = 720
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

    # 3. Dynamic Homography Mapper
    # Use calibrated attacking-box homography for Liverpool
    box_homo = DynamicPitchMapper.get_liverpool_attacking_box_homography()
    dyn_mapper = DynamicPitchMapper(anchor_homography=box_homo, smoothing_alpha=0.35)

    # Track pitch trajectory trails
    pitch_trajectories = defaultdict(lambda: deque(maxlen=trail_len))
    t0 = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        active_dets = frames_dict.get(frame_idx, [])
        num_players = len(active_dets)
        active_ids = set()

        # Update dynamic camera motion
        if use_dynamic_mapper:
            dyn_mapper.update_camera_motion(frame)

        # Copy fresh pitch background
        pitch_frame = base_pitch.copy()

        # Shot classification: is this tactical gameplay or a replay/close-up cutaway?
        # Replays and close-ups have fewer than min_tactical_players (e.g. 0-3 players)
        is_tactical = (num_players >= min_tactical_players)

        if is_tactical:
            # Project each active player
            projected_players = []
            for det in active_dets:
                tid = det["id"]
                x_p, y_p = det["x_pixel"], det["y_pixel"]

                if use_dynamic_mapper:
                    x_pitch, y_pitch, in_bounds = dyn_mapper.project_player(tid, x_p, y_p)
                else:
                    x_pitch = det["x_pitch_static"]
                    y_pitch = det["y_pitch_static"]
                    in_bounds = (0 <= x_pitch <= 105 and 0 <= y_pitch <= 68)

                if x_pitch is None or y_pitch is None:
                    continue

                active_ids.add(tid)
                cx = int(pad + x_pitch * scale_x)
                cy = int(pad + y_pitch * scale_y)
                cx = max(3, min(pitch_w - 4, cx))
                cy = max(3, min(pitch_h - 4, cy))
                pitch_trajectories[tid].append((cx, cy))

                projected_players.append({
                    "id": tid,
                    "cx": cx,
                    "cy": cy,
                    "in_bounds": in_bounds
                })

            # Draw trajectory trails on 2D tactical board
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

            # Draw player badges on 2D tactical board
            for p in projected_players:
                tid = p["id"]
                cx, cy = p["cx"], p["cy"]
                color = get_color(tid)

                if p["in_bounds"]:
                    cv2.circle(pitch_frame, (cx, cy), 9, color, -1)
                    cv2.circle(pitch_frame, (cx, cy), 10, (255, 255, 255), 1)
                    cv2.putText(pitch_frame, str(tid), (cx - 7, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.circle(pitch_frame, (cx, cy), 5, (160, 160, 160), -1)

            # Header on pitch canvas
            cv2.putText(pitch_frame, "2D TACTICAL RADAR (105m x 68m)", (pad, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(pitch_frame, f"Active Players: {len(projected_players)}", (pitch_w - 165, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        else:
            # Replay / Close-up / Non-tactical Cutaway banner
            pitch_trajectories.clear()
            banner_bg = pitch_frame[pitch_h//2 - 35 : pitch_h//2 + 35, 40 : pitch_w - 40]
            black_overlay = np.zeros(banner_bg.shape, dtype=np.uint8)
            pitch_frame[pitch_h//2 - 35 : pitch_h//2 + 35, 40 : pitch_w - 40] = cv2.addWeighted(banner_bg, 0.3, black_overlay, 0.7, 0)
            cv2.rectangle(pitch_frame, (40, pitch_h//2 - 35), (pitch_w - 40, pitch_h//2 + 35), (0, 200, 255), 1)

            msg1 = "REPLAY / CLOSE-UP CUTAWAY"
            msg2 = "Tactical Radar Paused"
            (w1, _), _ = cv2.getTextSize(msg1, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            (w2, _), _ = cv2.getTextSize(msg2, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(pitch_frame, msg1, (pitch_w//2 - w1//2, pitch_h//2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(pitch_frame, msg2, (pitch_w//2 - w2//2, pitch_h//2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

        # Composite video output
        if layout == "side_by_side":
            vid_resized = cv2.resize(frame, (vid_disp_w, vid_disp_h))
            pitch_resized = cv2.resize(pitch_frame, (disp_pitch_w, disp_pitch_h))
            combined = np.hstack([vid_resized, pitch_resized])
            out.write(combined)
        else:  # PIP
            pip_resized = cv2.resize(pitch_frame, (pip_w, pip_h))
            y_start = vid_h - pip_h - 15
            x_start = vid_w - pip_w - 15
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
    parser = argparse.ArgumentParser(description="Render dynamic 2D tactical radar video with camera motion compensation.")
    parser.add_argument("--video", required=True, help="Input video file path")
    parser.add_argument("--trajectories", required=True, help="Trajectory CSV with x_pixel, y_pixel or x1,y1,x2,y2")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--layout", default="side_by_side", choices=["side_by_side", "pip"], help="Display layout")
    args = parser.parse_args()

    render_radar_video(args.video, args.trajectories, args.output, layout=args.layout)

if __name__ == "__main__":
    main()
