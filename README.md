# Multi-Lane Traffic Density & Congestion Estimation (YOLOv8)

Real-time vehicle detection with YOLOv8, extended beyond a single
frame-wide vehicle count into **per-lane occupancy, congestion
classification, and speed estimation** — the kind of breakdown a
traffic engineer actually needs (which lane is backing up, not just
"there are 14 cars somewhere in frame").

![CI](https://github.com/stanley-metone/traffic-density-yolov8/actions/workflows/ci.yml/badge.svg)

## What this does differently from a plain vehicle counter

A single global count per frame tells you *that* traffic exists, not
*where* the problem is. This project instead:

1. **Zone-based analysis** — you define arbitrary polygonal regions
   (a lane, a junction box, a parking bay) over the camera frame.
   Each zone gets its own vehicle count, independent of the others.
2. **Density normalisation** — raw counts aren't comparable across
   zones of different sizes, so each zone's count is divided by an
   estimated *capacity* (derived from polygon area and an assumed
   average vehicle footprint, or set explicitly per zone).
3. **Discrete congestion classification** — `FREE_FLOW → LIGHT →
   MODERATE → HEAVY → GRIDLOCK`, driven by density **and** by tracked
   vehicle speed, so a half-empty lane full of stopped traffic still
   gets flagged correctly (density alone would miss this).
4. **Lightweight multi-object tracking** — a from-scratch centroid
   tracker (no DeepSORT/ByteTrack dependency) assigns stable IDs
   across frames so average speed per zone can be estimated from
   pixel displacement.

## Architecture

```
src/traffic_density/
  schemas.py     Detection / ZoneReport / FrameReport dataclasses
  zones.py       Polygon geometry: point-in-polygon, area, ZoneManager
  density.py     Occupancy ratio -> discrete congestion level
  tracker.py     Nearest-centroid multi-object tracker + speed estimate
  pipeline.py    Wires YOLOv8 + tracker + zones into a frame-by-frame report
  visualize.py   Colour-coded zone overlay for annotated output video
scripts/
  train.py       Fine-tune YOLOv8n on a vehicle dataset
  run_video.py   CLI: run the full pipeline over a video, write CSV (+ annotated mp4)
tests/           39 unit tests covering geometry, classification, and tracking logic
```

The geometry/classification/tracking core (`zones.py`, `density.py`,
`tracker.py`, `schemas.py`) is pure Python with **no** OpenCV/torch
dependency, so it's fully unit tested without a GPU or any model
weights. `pipeline.py` and `visualize.py` import `ultralytics`/`cv2`
lazily and are the only modules that need the ML stack installed.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run on a video with the stock COCO-pretrained model (car/truck/bus/motorcycle)
python scripts/run_video.py \
  --video data/my_intersection.mp4 \
  --model yolov8n.pt \
  --zones configs/zones.example.yaml \
  --out outputs/report.csv \
  --annotated-out outputs/annotated.mp4
```

`outputs/report.csv` gets one row per `(frame, zone)`:

| frame_index | timestamp_s | zone_name       | total_count | density_score | congestion_level | avg_speed_kmh |
|-------------|-------------|------------------|-------------|----------------|-------------------|----------------|
| 120         | 4.0         | lane_northbound  | 7           | 0.58           | MODERATE          | 22.4           |
| 120         | 4.0         | lane_southbound  | 2           | 0.17           | FREE_FLOW         | 41.1           |

## Defining your own zones

Zone polygons are just pixel coordinates on your source video's
resolution — pause a frame and click out the corners of each lane.
Edit `configs/zones.example.yaml`:

```yaml
zones:
  - name: lane_north
    polygon: [[180, 40], [420, 40], [460, 380], [120, 380]]
    # capacity omitted -> estimated from polygon area
  - name: lane_south
    polygon: [[460, 40], [700, 40], [780, 380], [500, 380]]
    capacity: 10   # or set explicitly if you know it
```

## Fine-tuning on a vehicle-specific dataset

The stock COCO-pretrained model works but misses vehicles in
oblique/overhead camera angles it wasn't trained on. See
[`data/README.md`](data/README.md) for the recommended dataset
(UAVDT aerial vehicles, via Roboflow) and:

```bash
python scripts/train.py --data data/vehicles/data.yaml --epochs 60 --name vehicle_finetune
```

## Quality checks

```bash
pip install -r requirements-dev.txt
pytest --cov=traffic_density tests/    # 39 tests, core logic
flake8 src tests scripts
black --check src tests scripts
mypy src
```

All of the above run in CI on every push (`.github/workflows/ci.yml`),
across Python 3.10–3.12.

## Design notes / trade-offs

- **Why a custom centroid tracker instead of ByteTrack/DeepSORT?**
  Fixed-camera traffic footage has much simpler motion than general
  MOT benchmarks — a greedy nearest-centroid match with a short grace
  period is enough, and keeping it dependency-free means the tracking
  logic is trivially unit-testable and has zero GPU requirement.
- **Why classify congestion from density *and* speed, not density
  alone?** A lane can be under-capacity but gridlocked (e.g. queuing
  behind a red light) — pure occupancy would under-report that.
- **Known limitation:** the tracker assumes non-overlapping,
  roughly-planar vehicle motion; it isn't tuned for dense highway
  merges with heavy occlusion. A production deployment would swap in
  ByteTrack for that case (the `pipeline.py` interface is written so
  that swap only touches `_build_zone_reports`).
