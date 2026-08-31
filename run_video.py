#!/usr/bin/env python3
"""CLI entry point: run the traffic-density pipeline over a video file.

Example
-------
    python scripts/run_video.py \\
        --video data/sample_intersection.mp4 \\
        --model runs/detect/vehicle_finetune/weights/best.pt \\
        --zones configs/zones.example.yaml \\
        --out outputs/report.csv \\
        --annotated-out outputs/annotated.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traffic_density.pipeline import TrafficDensityPipeline, write_reports_csv  # noqa: E402
from traffic_density.visualize import draw_zone_overlay  # noqa: E402
from traffic_density.zones import ZoneManager  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLOv8 weights (.pt)")
    parser.add_argument("--zones", required=True, help="Path to zones YAML config")
    parser.add_argument(
        "--out", default="outputs/report.csv", help="Where to write the CSV report"
    )
    parser.add_argument(
        "--annotated-out", default=None, help="Optional path to write an annotated video"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5, help="Detection confidence threshold"
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Process every Nth frame (speed vs. accuracy)",
    )
    parser.add_argument(
        "--px-per-meter", type=float, default=None, help="Calibration for km/h speed"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.zones) as f:
        zone_config = yaml.safe_load(f)
    zone_manager = ZoneManager.from_config(zone_config)

    pipeline = TrafficDensityPipeline(
        model_path=args.model,
        zone_manager=zone_manager,
        confidence=args.conf,
        px_per_meter=args.px_per_meter,
    )

    writer = None
    all_reports = []
    if args.annotated_out:
        import cv2

        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        Path(args.annotated_out).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.annotated_out, fourcc, fps, (width, height))

    if writer is not None:
        import cv2

        cap = cv2.VideoCapture(args.video)
        for report in pipeline.run(args.video, sample_every_n_frames=args.sample_every):
            all_reports.append(report)
            ok, frame = cap.read()
            if not ok:
                break
            draw_zone_overlay(frame, zone_manager, report.zones)
            writer.write(frame)
        cap.release()
        writer.release()
    else:
        for report in pipeline.run(args.video, sample_every_n_frames=args.sample_every):
            all_reports.append(report)
            for z in report.zones:
                print(
                    f"[frame {report.frame_index:>5}] {z.zone_name:<18} "
                    f"count={z.total_count:<3} density={z.density_score:<5} "
                    f"level={z.congestion_level}"
                )

    write_reports_csv(all_reports, args.out)
    print(f"\nWrote {len(all_reports)} frame reports to {args.out}")
    if args.annotated_out:
        print(f"Wrote annotated video to {args.annotated_out}")


if __name__ == "__main__":
    main()
