# Football Player Tracking & Performance Analytics Pipeline

An end-to-end computer vision pipeline for real-time football player detection, multi-object tracking (MOT), camera motion compensation, and trajectory analytics fine-tuned on the **SoccerNet Tracking-2023** dataset.

```
Raw Video Stream ──────► [Fine-Tuned YOLOv8s] ──────► [BoT-SORT + ORB GMC] ──────► [TrackEval / Analytics]
(1080p / 720p)             Detector (conf=0.35)        match=0.85, buffer=30         HOTA: 60.62 | IDF1: 70.62
```

---

## 1. Problem Statement

Multi-object tracking in association football broadcast footage presents unique computer vision challenges:
- **Teammate Visual Homogeneity**: Players on the same team share identical jerseys, shorts, and socks, confusing standard appearance-based Re-Identification (ReID) models.
- **Dynamic Camera Motion**: Continuous high-speed panning, tilting, and zooming induces rapid apparent player displacements that break standard linear motion assumptions.
- **Dense Crowd Occlusions**: Corners, free kicks, and penalty box scrums cause frequent multi-player bounding box overlaps ($\text{IoU} > 0.20$ in $> 45\%$ of active footage).
- **Broadcast Shot Transitions**: Frequent camera cuts, slow-motion replays, and touchline close-ups disrupt 2D coordinate spaces, risking false identity bleeding across scenes.

This pipeline systematically addresses each failure mode through detector fine-tuning, camera motion compensation (GMC), spatial association tuning, and camera-cut hygiene.

---

## 2. Dataset

The pipeline is trained and validated on the official **SoccerNet Tracking-2023** benchmark:
- **Sequences**: Broadcast match footage recorded at 25–30 FPS in 1080p and 720p.
- **Annotations**: MOT-format bounding boxes for players, referees, and goalkeepers.
- **Benchmark Split**: Evaluated across **12 validation sequences (9,000 frames)** containing over 148,000 ground-truth player instances.

---

## 3. Detection Architecture

- **Base Architecture**: YOLOv8s (*Small*)
- **Training Protocol**: 20 epochs, image size 640×640, batch size 2, PyTorch CUDA on NVIDIA RTX GPU.
- **Inference Confidence**: Optimized to `conf = 0.35` (empirically found to maximize association accuracy by filtering out low-confidence false positives while preserving 90.2% recall).
- **Detection Quality**:
  - **$\text{DetA}$**: `70.489`
  - **CLEAR Precision ($\text{CLR\_Pr}$)**: `96.812%`
  - **CLEAR Recall ($\text{CLR\_Re}$)**: `90.225%`
  - **$\text{MOTA}$**: `86.829`

---

## 4. Tracking Architecture

The pipeline uses **BoT-SORT** with dedicated football optimizations:
1. **Global Motion Compensation (GMC)**: Employs **ORB feature matching** to compute pitch homographies between successive frames, decoupling camera panning velocity from player sprint velocity.
2. **Kalman Filter State Estimation**: 8-dimensional state vector $[x, y, a, h, \dot{x}, \dot{y}, \dot{a}, \dot{h}]$ updated with camera motion compensation.
3. **Spatial Distance Gating**: High-tolerance spatial matching (`match_thresh = 0.85`), allowing the Kalman filter to bridge temporary occlusions and abrupt directional changes.
4. **Memory Buffer**: `track_buffer = 30 frames` ($\approx 1.0\text{–}1.2\text{s}$), allowing tracks to survive brief occlusions without retaining stale tracks across scene cuts.
5. **ReID Strategy**: Appearance ReID is explicitly disabled (`with_reid: False`). Because teammates wear identical kits, standard visual embeddings map distinct players to near-zero cosine distance, increasing identity swaps. Pure spatial-motion matching with ORB GMC delivers significantly higher association accuracy.

---

## 5. Evaluation Harness

All models are evaluated using the official SoccerNet MOT TrackEval harness ([`src/evaluation/run_trackeval.py`](src/evaluation/run_trackeval.py)) measuring:
- **$\text{HOTA}$** (*Higher Order Tracking Accuracy*): Balanced geometric mean of detection and association ($\sqrt{\text{DetA} \cdot \text{AssA}}$).
- **$\text{AssA}$** (*Association Accuracy*): Temporal alignment of player trajectories.
- **$\text{DetA}$** (*Detection Accuracy*): Spatial alignment of bounding boxes.
- **$\text{IDF1}$**: Identity F1 score measuring global trajectory consistency.
- **$\text{IDSW}$**: Total identity switches across the benchmark.
- **$\text{Frag}$**: Trajectory fragmentations.

---

## 6. Systematic Experimentation & Progression

18 systematic experiments were conducted across parameter sweeps on the 12 validation sequences:

| Experiment | Configuration | HOTA | AssA | IDF1 | IDSW | Frag | DetA | MOTA |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | `conf=0.20`, `sparseOptFlow`, `match=0.80` | 59.863 | 50.978 | 68.228 | 945 | 1542 | 70.429 | 86.547 |
| **Exp 4: Buffer** | `track_buffer = 15` | 59.765 | 50.864 | 68.199 | 930 | 1535 | 70.356 | 86.459 |
| **Exp 4: Buffer** | `track_buffer = 60` | 59.816 | 50.958 | 68.258 | 975 | 1579 | 70.346 | 86.566 |
| **Exp 5: Conf** | `conf = 0.15` | 59.763 | 50.890 | 68.038 | 936 | 1507 | 70.316 | 86.366 |
| **Exp 5: Conf** | `conf = 0.30` | 59.924 | 51.147 | 68.864 | 881 | 1600 | 70.331 | 86.696 |
| **Exp 5: Conf** | `conf = 0.35` | 60.368 | 51.753 | 69.569 | 728 | 1618 | 70.529 | 86.801 |
| **Exp 6: GMC** | `gmc = none` (No GMC) | 57.120 | 47.112 | 64.275 | 1099 | 1520 | 69.432 | 85.382 |
| **Exp 6: GMC** | `gmc = ecc` | 58.769 | 49.245 | 66.854 | 978 | 1510 | 70.278 | 86.344 |
| **Exp 6: GMC** | `gmc = orb` | 60.109 | 51.387 | 69.185 | 889 | 1536 | 70.425 | 86.490 |
| **Exp 7: Champion** | `conf=0.35 + ORB (match=0.80)` | **60.652** | **52.230** | 70.289 | 720 | 1585 | **70.532** | 86.799 |
| **Exp 8: Match** | `conf=0.35 + ORB + match=0.75` | 60.094 | 51.326 | 69.135 | 811 | 1601 | 70.480 | 86.699 |
| **Exp 8: Match** | `conf=0.35 + ORB + match=0.85` | **60.618** | **52.204** | **70.615** | **693** | 1605 | 70.489 | **86.829** |
| **Exp 9: ReID** | `conf=0.35 + ORB + match=0.85 + ReID=True` | 59.402 | 50.173 | 68.911 | 815 | 1660 | 70.440 | 86.683 |

---

## 7. Official Benchmark Results

### Official Final Configuration (`experiments/final/botsort_champion.yaml`)
- **Detector**: YOLOv8s @ `conf = 0.35`
- **Tracker**: BoT-SORT, `GMC = ORB`, `match_thresh = 0.85`, `track_buffer = 30`, `with_reid = False`
- **Official Scores**:
  - **HOTA**: `60.618`
  - **DetA**: `70.489`
  - **AssA**: `52.204`
  - **IDF1**: `70.615` *(Peak across all experiments)*
  - **MOTA**: `86.829` *(Peak across all experiments)*
  - **IDSW**: `693` *(-252 ID switches vs. baseline)*
  - **Frag**: `1605`

### HOTA-Oriented Alternative (`experiments/botsort_orb/botsort_conf35_orb.yaml`)
- Same as final configuration with `match_thresh = 0.80`.
- **HOTA**: `60.652` | **AssA**: `52.230` | **IDF1**: `70.289` | **IDSW**: `720` | **Frag**: `1585`.

---

## 8. Real-World Video Stress-Test Evaluations

The pipeline was evaluated across 5 full real-world match videos testing different failure modes:

| Test Video | Stress Factor | Resolution & FPS | Duration | Active Tracks/Frame | Cuts | Cross-Cut Bleed | Occlusion Gaps Bridged | Mean Continuity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `penalty_box_crowded.mp4` | Heavy scrums / box clustering | 1080p @ 25 FPS | 30.0s | 20.93 | 0 | 0.0% | 119 | 92.0% |
| `camera_motion_counterattack.mp4` | Rapid pans / fast transitions | 1080p @ 25 FPS | 30.0s | 13.89 | 0 | 0.0% | 18 | 98.3% |
| `liverpool_highlights.mp4` | Replays, cuts, close-ups | 720p @ 30 FPS | 140.5s | 14.64 | 10 | **0.0%** | 729 | 89.4% |
| `barcelona_highlight.mp4` | Sidelines, substitutions, whip-pans | 720p @ 25 FPS | 179.6s | 11.16 | 25 | **0.0%** | 720 | 87.8% |
| `highlight.mp4` | Continuous long tactical feed | 1080p @ 30 FPS | 121.1s | **23.19** | 0 | 0.0% | **1,253** | 84.2% |

For each video, four standardized artifacts are automatically generated in `outputs/evaluation/videos/<video_name>/`:
- `video_metrics.json`: Full scene dynamics, cut counts, occlusion volume, and continuity stats.
- `tracking.csv`: Raw frame-by-frame detections, bounding box coordinates, and track IDs.
- `annotated.mp4`: High-speed rendered visual output with bounding boxes, `#id` badges, foot ground-anchors, 25-frame trajectory trails, and a real-time HUD dashboard.
- `track_statistics.csv`: Individual track duration, span, and confidence statistics.

---

## 9. Repository Structure

```
football-player-tracking/
│
├── data/
│   ├── raw/               # Raw SoccerNet tracking-2023 dataset
│   ├── processed/         # Converted YOLO-format labels & images
│   └── videos/            # Evaluation match clips (MP4)
│
├── models/                # Model weights directory
│
├── src/
│   ├── data/              # Conversion & dataset validation scripts
│   ├── detection/         # YOLO inference & test scripts
│   ├── tracking/          # BoT-SORT runners, evaluations, and renderers
│   ├── evaluation/        # TrackEval harness, summaries, and comparisons
│   └── visualization/     # Standalone video visualizer with HUD & trails
│
├── experiments/
│   ├── baseline/          # Baseline BoT-SORT YAML (conf=0.20, sparseOptFlow)
│   ├── botsort_orb/       # HOTA-oriented YAML (conf=0.35, ORB, match=0.80)
│   └── final/             # Official champion YAML (conf=0.35, ORB, match=0.85)
│
├── outputs/
│   ├── tracking/          # Tracking prediction CSVs
│   ├── evaluation/        # Benchmark CSVs, JSON records, and video dynamics
│   └── videos/            # Rendered annotated MP4 videos
│
├── docs/
│   └── detection_vs_tracking.md  # Analytical decomposition (DetA vs AssA)
│
├── notebooks/             # Exploratory analysis notebooks
├── requirements.txt       # Python package dependencies
├── LICENSE                # MIT License
└── README.md              # Project documentation
```

---

## 10. Quick Start & Reproducibility

### Installation
```bash
git clone https://github.com/DA-Shaurya/football-player-tracking.git
cd football-player-tracking
python -m venv .venv
.venv\Scripts\activate      # Windows (or source .venv/bin/activate on Linux)
pip install -r requirements.txt
```

### Run TrackEval Benchmark
To reproduce official benchmark metrics on the SoccerNet validation set:
```bash
python src/evaluation/run_trackeval.py --experiment botsort_conf35_orb_match85
```

### Compare Experiments Side-by-Side
```bash
python src/evaluation/compare_experiments.py --baseline botsort --candidates botsort_conf35_orb botsort_conf35_orb_match85 botsort_conf35_orb_match85_reid
```

### Evaluate Any Real-World Video
To generate the 4 standard artifacts (`tracking.csv`, `annotated.mp4`, `video_metrics.json`, `track_statistics.csv`) on any video:
```bash
python src/tracking/evaluate_video.py --video data/videos/liverpool_highlights.mp4
```

### Render Video Visualization with HUD & Trails
```bash
python src/visualization/render_tracking.py --video data/videos/highlight.mp4 --csv outputs/tracking/highlight_tracking_champion.csv --output outputs/videos/highlight_annotated.mp4 --title "Premier League Tactical Feed"
```

---

## 11. Limitations & Future Work

### Limitations
1. **Prolonged Scrums**: Pure spatial matching can switch identities when two opposing players tussle and separate in swapped positions if occlusion exceeds 30 frames.
2. **Camera Pans during Extreme Whip-Motion**: High-velocity panning causes motion blur, occasionally dropping confidence below 0.35 before ORB GMC re-acquires features.
3. **Absence of Player Identity Across Scene Cuts**: By design, tracks cleanly terminate at camera cuts to prevent false associations, meaning cross-shot player re-identification requires higher-level visual understanding.

### Future Work
1. **Jersey Number Optical Character Recognition (OCR)**: Integrating an OCR sub-network on back crops to re-anchor player identity across camera cuts.
2. **Team Jersey Color Clustering**: Unsupervised HSV/CIELAB color clustering to prevent teammate-vs-opponent false associations during scrums.
3. **Pitch Coordinate Homography**: Projecting bounding box feet coordinates onto a static 2D bird's-eye pitch model for sports tactical analytics.

---

## License

This project is licensed under the [MIT License](LICENSE).
