"""Polygonal zone definitions (lanes, junction arms, etc.) and the
logic that assigns detections to zones.

This is the piece that makes the project "multi-lane" rather than a
single global vehicle counter: each zone is an independent polygon
with its own pixel area, so density/congestion is reported per lane,
not just for the whole frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

Point = tuple[float, float]


def _shoelace_area(polygon: list[Point]) -> float:
    """Absolute polygon area via the shoelace formula."""
    n = len(polygon)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    """Ray-casting point-in-polygon test.

    Standard even-odd rule. O(n) in the number of polygon vertices;
    fine for the small (4-8 vertex) zones this project uses.
    """
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


@dataclass
class Zone:
    """A named polygonal region of interest within the frame.

    Parameters
    ----------
    name:
        Human readable identifier, e.g. ``"lane_north"``.
    polygon:
        List of ``(x, y)`` pixel-coordinate vertices, in order
        (clockwise or counter-clockwise both work).
    capacity:
        Approximate number of vehicles the zone can hold before it is
        considered saturated. Used to normalise the density score.
        If omitted, it is estimated from the polygon area assuming an
        average vehicle footprint of ``avg_vehicle_area_px``.
    avg_vehicle_area_px:
        Assumed on-screen area (in px^2) of a single vehicle, used
        only to estimate ``capacity`` when it isn't provided
        explicitly. Tune this per-camera (closer camera -> larger
        value).
    """

    name: str
    polygon: list[Point]
    capacity: int | None = None
    avg_vehicle_area_px: float = 4500.0

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ValueError(
                f"Zone '{self.name}' needs at least 3 vertices, got {len(self.polygon)}"
            )
        if self.capacity is None:
            area = self.area
            estimated = max(1, round(area / self.avg_vehicle_area_px))
            self.capacity = estimated

    @property
    def area(self) -> float:
        return _shoelace_area(self.polygon)

    def contains(self, point: Point) -> bool:
        return point_in_polygon(point, self.polygon)


@dataclass
class ZoneManager:
    """Holds a set of :class:`Zone` objects and assigns points to them."""

    zones: list[Zone] = field(default_factory=list)

    def add(self, zone: Zone) -> None:
        if any(z.name == zone.name for z in self.zones):
            raise ValueError(f"Duplicate zone name: {zone.name}")
        self.zones.append(zone)

    def zone_for_point(self, point: Point) -> Zone | None:
        """Return the first zone containing ``point``, or ``None``.

        Zones are assumed non-overlapping for this project's use case
        (adjacent traffic lanes); if zones do overlap, the first match
        in insertion order wins.
        """
        for zone in self.zones:
            if zone.contains(point):
                return zone
        return None

    def get(self, name: str) -> Zone:
        for zone in self.zones:
            if zone.name == name:
                return zone
        raise KeyError(f"No such zone: {name}")

    @classmethod
    def from_config(cls, config: dict) -> "ZoneManager":
        """Build a :class:`ZoneManager` from a parsed YAML/JSON config.

        Expected shape::

            zones:
              - name: lane_north
                polygon: [[100, 50], [300, 50], [320, 400], [80, 400]]
                capacity: 12   # optional
        """
        manager = cls()
        for entry in config.get("zones", []):
            polygon = [tuple(p) for p in entry["polygon"]]
            manager.add(
                Zone(
                    name=entry["name"],
                    polygon=polygon,
                    capacity=entry.get("capacity"),
                    avg_vehicle_area_px=entry.get("avg_vehicle_area_px", 4500.0),
                )
            )
        return manager
