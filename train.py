#!/usr/bin/env python3
"""Fine-tune a YOLOv8 nano model on a vehicle-detection dataset.

This script is intentionally thin -- Ultralytics already provides a
solid training loop; the value-add here is a reproducible, versioned
entry point with sane defaults for this project (see README ->
"Dataset" for how to obtain/format the data).

Example
-------
    python scripts/train.py \\
        --data data/vehicles/data.yaml \\
        --epochs 60 \\
        --imgsz 640 \\
        --name vehicle_finetune
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to YOLO-format data.yaml")
    parser.add_argument(
        "--base-model", default="yolov8n.pt", help="Pretrained weights to start from"
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--name", default="vehicle_finetune", help="Run name under runs/detect/"
    )
    parser.add_argument("--patience", type=int, default=15, help="Early-stopping patience")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(args.base_model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        patience=args.patience,
    )
    metrics = model.val()
    print("Validation metrics:", metrics.results_dict)


if __name__ == "__main__":
    main()
