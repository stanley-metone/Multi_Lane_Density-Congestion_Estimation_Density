"""Drawing helpers for overlaying zones, counts, and congestion colour
coding onto video frames. Imports OpenCV lazily for the same reason
as :mod:`traffic_density.pipeline`.
"""

from __future__ import annotations

from .density import CongestionLevel
from .schemas import ZoneReport
from .zones import ZoneManager

_LEVEL_COLOR_BGR = {
    CongestionLevel.FREE_FLOW.value: (0, 200, 0),
    CongestionLevel.LIGHT.value: (0, 220, 220),
    CongestionLevel.MODERATE.value: (0, 165, 255),
    CongestionLevel.HEAVY.value: (0, 80, 255),
    CongestionLevel.GRIDLOCK.value: (0, 0, 255),
}


def draw_zone_overlay(frame, zone_manager: ZoneManager, zone_reports: list[ZoneReport]):
    """Draw each zone's polygon, colour-coded by congestion level, plus
    a label with vehicle count and congestion level. Returns the
    (mutated) frame for convenience.
    """
    import cv2
    import numpy as np

    report_by_zone = {r.zone_name: r for r in zone_reports}

    for zone in zone_manager.zones:
        report = report_by_zone.get(zone.name)
        level = report.congestion_level if report else CongestionLevel.FREE_FLOW.value
        color = _LEVEL_COLOR_BGR.get(level, (255, 255, 255))

        pts = np.array(zone.polygon, dtype=np.int32).reshape((-1, 1, 2))
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, dst=frame)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

        label_pt = zone.polygon[0]
        count = report.total_count if report else 0
        label = f"{zone.name}: {count} ({level})"
        cv2.putText(
            frame,
            label,
            (int(label_pt[0]), max(15, int(label_pt[1]) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame
