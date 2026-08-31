"""End-to-end video processing pipeline.

Wires together YOLOv8 detection, the :class:`CentroidTracker`, and
:class:`ZoneManager` to produce a per-frame :class:`FrameReport`.

This module imports ``ultralytics`` and ``cv2`` lazily (inside
``__init__``/``run``) rather than at module scope, so the rest of the
package -- and its unit tests -- can be imported without those heavy,
GPU-oriented dependencies installed.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Iterable, Optional

from .density import classify_congestion, compute_density
from .schemas import Detection, FrameReport, ZoneReport
from .tracker import CentroidTracker
from .zones import ZoneManager

logger = logging.getLogger(__name__)

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}


class TrafficDensityPipeline:
    """Runs YOLOv8 inference over a video and reports zone-level density.

    Parameters
    ----------
    model_path:
        Path to a ``.pt`` YOLOv8 weights file (pretrained or your own
        fine-tuned model).
    zone_manager:
        Pre-built :class:`ZoneManager` describing the lanes/regions to
        monitor for this camera's framing.
    confidence:
        Minimum detection confidence to keep.
    fps_override:
        If set, used instead of the video's own reported FPS when
        converting tracker speed (px/frame) to px/second. Useful when
        a source video's metadata FPS is unreliable.
    px_per_meter:
        Optional calibration factor (pixels per real-world metre at
        the camera's ground plane) to additionally report speed in
        km/h. Leave as ``None`` to skip km/h conversion.
    """

    def __init__(
        self,
        model_path: str,
        zone_manager: ZoneManager,
        confidence: float = 0.5,
        fps_override: Optional[float] = None,
        px_per_meter: Optional[float] = None,
    ):
        self.model_path = model_path
        self.zone_manager = zone_manager
        self.confidence = confidence
        self.fps_override = fps_override
        self.px_per_meter = px_per_meter
        self._model = None  # lazy-loaded

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO  # local import: heavy, GPU-oriented

            logger.info("Loading YOLO model from %s", self.model_path)
            self._model = YOLO(self.model_path)
        return self._model

    def _detect(self, frame) -> list[Detection]:
        model = self._load_model()
        results = model.predict(source=frame, conf=self.confidence, verbose=False)[0]
        detections: list[Detection] = []
        names = results.names
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id, str(cls_id))
            if cls_name not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            conf = float(box.conf[0])
            detections.append(
                Detection(x1=x1, y1=y1, x2=x2, y2=y2, class_name=cls_name, confidence=conf)
            )
        return detections

    def run(self, video_path: str, sample_every_n_frames: int = 1) -> Iterable[FrameReport]:
        """Yield a :class:`FrameReport` for each processed frame.

        A generator so callers can stream results to disk/stdout
        without holding the whole video's reports in memory.
        """
        import cv2  # local import, see module docstring

        tracker = CentroidTracker()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        fps = self.fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_index = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % sample_every_n_frames != 0:
                    frame_index += 1
                    continue

                detections = self._detect(frame)
                centroids = [d.centroid for d in detections]
                track_ids = tracker.update(centroids, frame_index)

                zone_reports = self._build_zone_reports(
                    detections, track_ids, tracker, fps, frame_index
                )
                yield FrameReport(
                    frame_index=frame_index,
                    timestamp_s=frame_index / fps,
                    zones=zone_reports,
                )
                frame_index += 1
        finally:
            cap.release()

    def _build_zone_reports(
        self,
        detections: list[Detection],
        track_ids: list[int],
        tracker: CentroidTracker,
        fps: float,
        frame_index: int,
    ) -> list[ZoneReport]:
        per_zone_counts: dict[str, dict[str, int]] = {
            z.name: {} for z in self.zone_manager.zones
        }
        per_zone_speeds: dict[str, list[float]] = {z.name: [] for z in self.zone_manager.zones}

        for det, tid in zip(detections, track_ids):
            zone = self.zone_manager.zone_for_point(det.centroid)
            if zone is None:
                continue
            per_zone_counts[zone.name][det.class_name] = (
                per_zone_counts[zone.name].get(det.class_name, 0) + 1
            )
            speed_px_frame = tracker.speed_px_per_frame(tid)
            if speed_px_frame is not None:
                per_zone_speeds[zone.name].append(speed_px_frame * fps)

        reports = []
        for zone in self.zone_manager.zones:
            counts = per_zone_counts[zone.name]
            total = sum(counts.values())
            speeds = per_zone_speeds[zone.name]
            avg_speed_px_s = sum(speeds) / len(speeds) if speeds else None
            avg_speed_kmh = (
                (avg_speed_px_s / self.px_per_meter) * 3.6
                if avg_speed_px_s is not None and self.px_per_meter
                else None
            )
            density = compute_density(total, zone.capacity)
            level = classify_congestion(density, avg_speed_px_s)
            reports.append(
                ZoneReport(
                    zone_name=zone.name,
                    frame_index=frame_index,
                    counts_by_class=counts,
                    total_count=total,
                    density_score=round(density, 4),
                    congestion_level=level.value,
                    avg_speed_px_s=round(avg_speed_px_s, 2) if avg_speed_px_s else None,
                    avg_speed_kmh=round(avg_speed_kmh, 2) if avg_speed_kmh else None,
                )
            )
        return reports


def write_reports_csv(reports: Iterable[FrameReport], out_path: str) -> None:
    """Flatten frame/zone reports into a tidy per-(frame, zone) CSV."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_index",
        "timestamp_s",
        "zone_name",
        "total_count",
        "density_score",
        "congestion_level",
        "avg_speed_px_s",
        "avg_speed_kmh",
        "counts_by_class",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            for zone in report.zones:
                writer.writerow(
                    {
                        "frame_index": report.frame_index,
                        "timestamp_s": round(report.timestamp_s, 3),
                        "zone_name": zone.zone_name,
                        "total_count": zone.total_count,
                        "density_score": zone.density_score,
                        "congestion_level": zone.congestion_level,
                        "avg_speed_px_s": zone.avg_speed_px_s,
                        "avg_speed_kmh": zone.avg_speed_kmh,
                        "counts_by_class": json.dumps(zone.counts_by_class),
                    }
                )
