"""Density scoring and discrete congestion classification.

This is the analytical core of the project: turning a raw vehicle
count for a zone into (a) a normalised density score and (b) a
human-meaningful congestion label, using both occupancy (count vs
capacity) and, when available, average vehicle speed -- a slow-moving
but not-yet-full lane is still worth flagging as congested.
"""

from __future__ import annotations

from enum import Enum


class CongestionLevel(str, Enum):
    FREE_FLOW = "FREE_FLOW"
    LIGHT = "LIGHT"
    MODERATE = "MODERATE"
    HEAVY = "HEAVY"
    GRIDLOCK = "GRIDLOCK"


# Density-score thresholds (count / capacity) -> level, checked in order.
_DENSITY_THRESHOLDS: list[tuple[float, CongestionLevel]] = [
    (0.20, CongestionLevel.FREE_FLOW),
    (0.45, CongestionLevel.LIGHT),
    (0.70, CongestionLevel.MODERATE),
    (0.90, CongestionLevel.HEAVY),
]
_MAX_LEVEL = CongestionLevel.GRIDLOCK

# Below this average speed (px/s), a lane is bumped up one congestion
# level even if occupancy alone wouldn't justify it -- stopped traffic
# in a half-empty lane is still a problem.
_SLOW_SPEED_PX_S = 15.0

_LEVEL_ORDER = [
    CongestionLevel.FREE_FLOW,
    CongestionLevel.LIGHT,
    CongestionLevel.MODERATE,
    CongestionLevel.HEAVY,
    CongestionLevel.GRIDLOCK,
]


def compute_density(vehicle_count: int, capacity: int) -> float:
    """Occupancy ratio, clamped to ``[0, 1]``.

    ``capacity`` is the estimated max vehicles a zone can hold (see
    :class:`traffic_density.zones.Zone`). A ``capacity`` of 0 is
    treated as 1 to avoid division errors, with a warning left to the
    caller's logging rather than raised here (keeps this a pure,
    exception-free function for the hot per-frame path).
    """
    safe_capacity = max(1, capacity)
    ratio = vehicle_count / safe_capacity
    return max(0.0, min(1.0, ratio))


def classify_congestion(
    density_score: float,
    avg_speed_px_s: float | None = None,
) -> CongestionLevel:
    """Map a density score (+ optional speed) to a :class:`CongestionLevel`.

    Parameters
    ----------
    density_score:
        Value in ``[0, 1]`` from :func:`compute_density`.
    avg_speed_px_s:
        Optional average tracked-vehicle speed in pixels/second for
        the zone this frame. When provided and below
        ``_SLOW_SPEED_PX_S``, the computed level is escalated by one
        step (capped at GRIDLOCK) to reflect stopped/crawling traffic.
    """
    level = _MAX_LEVEL
    for threshold, candidate in _DENSITY_THRESHOLDS:
        if density_score <= threshold:
            level = candidate
            break

    if avg_speed_px_s is not None and avg_speed_px_s < _SLOW_SPEED_PX_S:
        idx = _LEVEL_ORDER.index(level)
        level = _LEVEL_ORDER[min(idx + 1, len(_LEVEL_ORDER) - 1)]

    return level
