"""Typed data structures shared across the pipeline.

Kept dependency-free (stdlib ``dataclasses`` only) so the rest of the
package can be imported and unit tested without pulling in
ultralytics/opencv/torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Detection:
    """A single vehicle detection in image-pixel coordinates.

    ``x1, y1, x2, y2`` are the bounding box corners. ``track_id`` is
    optional and only populated once the detection has passed through
    the :class:`~traffic_density.tracker.CentroidTracker`.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str
    confidence: float
    track_id: Optional[int] = None

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass
class ZoneReport:
    """Per-zone summary for a single frame."""

    zone_name: str
    frame_index: int
    counts_by_class: dict[str, int]
    total_count: int
    density_score: float
    congestion_level: str
    avg_speed_px_s: Optional[float] = None
    avg_speed_kmh: Optional[float] = None


@dataclass
class FrameReport:
    """All zone reports for a single processed frame, plus metadata."""

    frame_index: int
    timestamp_s: float
    zones: list[ZoneReport] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "zones": [z.__dict__ for z in self.zones],
        }
