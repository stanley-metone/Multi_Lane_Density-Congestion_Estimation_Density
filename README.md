# Dataset

This project is dataset-agnostic (any YOLO-format vehicle-detection
dataset works with `scripts/train.py`), but it was built and
validated against:

**[UAVDT Aerial Vehicles Dataset](https://universe.roboflow.com/uavdt/aerial-vehicles-hjarh)**
(Roboflow Universe) — overhead/drone imagery of roads and
intersections, annotated for `car` / `truck` / `bus` classes, exportable
directly in YOLOv8 format.

An alternative, larger benchmark if you want more scale or want to
compare against published numbers is
**[UA-DETRAC](https://detrac-db.rit.albany.edu/)**, a widely-cited
surveillance-camera vehicle dataset (10 hours of video, 140k frames,
8,250 annotated vehicles) — useful if your target camera is a
street-level/CCTV angle rather than overhead.

## Getting the data

1. Create a free [Roboflow](https://roboflow.com/) account.
2. Open the UAVDT Aerial Vehicles dataset link above, click **Download
   Dataset**, choose the **YOLOv8** export format, and select
   "download zip to computer" (or use the generated `roboflow` pip
   snippet to pull it straight into `data/`).
3. Unzip into `data/vehicles/` so you end up with:

   ```
   data/vehicles/
     data.yaml
     train/images  train/labels
     valid/images  valid/labels
     test/images   test/labels
   ```

4. Point `scripts/train.py --data` at `data/vehicles/data.yaml`.

## Using your own footage

For the *inference* side (density estimation on a specific
intersection), you don't need a labelled dataset at all — just a
video file of that camera's feed. Use `scripts/run_video.py` with
either the stock `yolov8n.pt` weights or your fine-tuned model. The
zone polygons in `configs/zones.example.yaml` are specific to one
camera framing; redraw them for yours (see the README's "Defining
your own zones" section).
