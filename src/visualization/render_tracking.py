import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict, deque
import cv2
import numpy as np
import pandas as pd

def get_color(track_id):
    """Generate a consistent, vibrant BGR color for each track ID."""
    np.random.seed(int(track_id) * 31 + 17)
    hue = np.random.randint(0, 180)
    sat = np.random.randint(180, 255)
    val = np.random.randint(200, 255)
    hsv_pixel = np.uint8([[[hue, sat, val]]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2]))

def render_annotated_video(
    video_path,
    csv_path,
    output_path,
    match_title="Football Player Tracking",
    trail_len=25,
    show_fps=True
):
    video_path = Path(video_path)
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PRODUCING TRACKED FOOTBALL VIDEO VISUALIZATION")
    print(f"Match/Title: {match_title}")
    print(f"Source: {video_path}")
    print(f"Tracking Data: {csv_path}")
    print(f"Output: {output_path}")
    print("=" * 70)

    # 1. Load Tracking CSV
    df = pd.read_csv(csv_path)
    frames_dict = defaultdict(list)
    for _, row in df.iterrows():
        frames_dict[int(row["frame"])].append({
            "id": int(row["track_id"]),
            "conf": float(row["confidence"]),
            "x1": int(round(row["x1"])),
            "y1": int(round(row["y1"])),
            "x2": int(round(row["x2"])),
            "y2": int(round(row["y2"]))
        })

    # 2. Open Video Capture & Writer
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps): fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")

    trajectories = defaultdict(lambda: deque(maxlen=trail_len))
    t0 = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        detections = frames_dict.get(frame_idx, [])
        active_ids = set()

        # Update trajectory points
        for det in detections:
            tid = det["id"]
            active_ids.add(tid)
            feet_x = int((det["x1"] + det["x2"]) / 2)
            feet_y = det["y2"]
            trajectories[tid].append((feet_x, feet_y))

        # Draw trajectory trails
        for tid, points in list(trajectories.items()):
            if tid not in active_ids and len(points) > 0:
                points.popleft()
                if len(points) == 0:
                    del trajectories[tid]
                    continue

            color = get_color(tid)
            pts_list = list(points)
            for i in range(1, len(pts_list)):
                alpha = i / len(pts_list)
                thickness = max(1, int(round(1 + 2 * alpha)))
                cv2.line(frame, pts_list[i - 1], pts_list[i], color, thickness)

        # Draw bounding boxes, anchor circles, and labels
        for det in detections:
            tid = det["id"]
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            conf = det["conf"]
            color = get_color(tid)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Foot ground contact point
            feet_x = int((x1 + x2) / 2)
            cv2.circle(frame, (feet_x, y2), 3, color, -1)

            # Label badge (#id)
            label = f"#{tid}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y1 = max(0, y1 - th - 6)
            label_y2 = y1
            label_x2 = x1 + tw + 8

            cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        # Modern HUD Dashboard Overlay (Top-Left)
        hud_w = 400
        hud_h = 90
        sub_img = frame[10:10+hud_h, 10:10+hud_w]
        black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
        res = cv2.addWeighted(sub_img, 0.30, black_rect, 0.70, 1.0)
        frame[10:10+hud_h, 10:10+hud_w] = res
        cv2.rectangle(frame, (10, 10), (10+hud_w, 10+hud_h), (255, 255, 255), 1)

        elapsed_so_far = time.time() - t0
        curr_fps = frame_idx / elapsed_so_far if elapsed_so_far > 0 else fps
        sec = frame_idx / fps
        tot_sec = total_frames / fps

        line1 = f"{match_title}"
        line2 = f"Time: {int(sec//60):02d}:{sec%60:04.1f} / {int(tot_sec//60):02d}:{tot_sec%60:04.1f}  (Frame {frame_idx}/{total_frames})"
        line3 = f"Active Players: {len(detections):<3} | FPS: {curr_fps:.1f} | Tracker: BoT-SORT"

        cv2.putText(frame, line1, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, line2, (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, line3, (18, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1, cv2.LINE_AA)

        out.write(frame)

        if frame_idx % 500 == 0 or frame_idx == total_frames:
            print(f"  Rendered {frame_idx}/{total_frames} frames ({curr_fps:.1f} FPS)...")

    cap.release()
    out.release()
    total_time = time.time() - t0
    print(f"\nRendered video saved to {output_path} ({total_time:.1f}s, {frame_idx/total_time:.1f} FPS)")

def main():
    parser = argparse.ArgumentParser(description="Render football tracking video with trails and HUD.")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--csv", required=True, help="Path to tracking predictions CSV")
    parser.add_argument("--output", required=True, help="Path to output MP4 video")
    parser.add_argument("--title", default="Football Player Tracking", help="Match title for HUD")
    parser.add_argument("--trail", type=int, default=25, help="Length of trajectory trail")
    args = parser.parse_args()

    render_annotated_video(args.video, args.csv, args.output, match_title=args.title, trail_len=args.trail)

if __name__ == "__main__":
    main()
