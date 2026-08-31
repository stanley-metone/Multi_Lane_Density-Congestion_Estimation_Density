"""
traffic_density
================

A multi-lane, real-time traffic density and congestion estimation
toolkit built on top of Ultralytics YOLOv8.

Unlike a plain "count every vehicle in the frame" approach, this
package lets you define arbitrary polygonal zones (lanes, junction
arms, parking bays, ...) over a video feed and reports, per zone and
per frame:

* vehicle counts by class (car / truck / bus / motorcycle)
* a density score normalised by the zone's pixel area
* a discrete congestion level (FREE_FLOW -> GRIDLOCK)
* approximate average vehicle speed (pixels/sec, optionally
  calibrated to km/h) using centroid tracking across frames

The core logic (geometry, classification thresholds, tracking) is
pure Python with no video/model dependency, so it is fully unit
tested without needing a GPU, a video file, or model weights.
"""

from .zones import Zone, ZoneManager
from .density import CongestionLevel, classify_congestion, compute_density
from .tracker import CentroidTracker
from .schemas import Detection, FrameReport, ZoneReport

__all__ = [
    "Zone",
    "ZoneManager",
    "CongestionLevel",
    "classify_congestion",
    "compute_density",
    "CentroidTracker",
    "Detection",
    "FrameReport",
    "ZoneReport",
]

__version__ = "0.1.0"
