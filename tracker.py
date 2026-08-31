"""A minimal centroid tracker.

YOLOv8 gives per-frame detections with no identity across frames. To
report vehicle speed we need to know "this car in frame N is the same
car as that one in frame N-1". Rather than pulling in a full
DeepSORT/ByteTrack dependency, this is a small, dependency-free
nearest-centroid tracker with a disappearance grace period -- more
than adequate for the fixed, semi-overhead traffic-camera framing
this project targets, and easy to unit test in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

Point = tuple[float, float]


@dataclass
class _Track:
    track_id: int
    centroid: Point
    history: list[tuple[Point, int]] = field(default_factory=list)  # (centroid, frame_idx)
    frames_since_seen: int = 0


class CentroidTracker:
    """Assigns stable IDs to detections across frames by nearest centroid.

    Parameters
    ----------
    max_distance_px:
        Maximum centroid displacement between consecutive frames for
        two detections to be considered the same object. Should scale
        with expected vehicle speed and frame rate.
    max_missed_frames:
        Number of consecutive frames a track may go unmatched before
        it is dropped (handles brief detector misses/occlusion).
    """

    def __init__(self, max_distance_px: float = 80.0, max_missed_frames: int = 5):
        self.max_distance_px = max_distance_px
        self.max_missed_frames = max_missed_frames
        self._tracks: dict[int, _Track] = {}
        self._next_id = 0

    @property
    def active_track_ids(self) -> list[int]:
        return list(self._tracks.keys())

    def update(self, centroids: list[Point], frame_index: int) -> list[int]:
        """Match ``centroids`` (this frame's detections) to existing tracks.

        Returns a list of track IDs, one per input centroid, in the
        same order as ``centroids``. New objects get freshly minted
        IDs; unmatched existing tracks age and are dropped after
        ``max_missed_frames``.
        """
        assigned: list[int | None] = [None] * len(centroids)
        available_track_ids = set(self._tracks.keys())

        # Greedy nearest-neighbour matching, closest pairs first.
        candidate_pairs = []
        for i, c in enumerate(centroids):
            for tid in available_track_ids:
                d = _euclidean(c, self._tracks[tid].centroid)
                if d <= self.max_distance_px:
                    candidate_pairs.append((d, i, tid))
        candidate_pairs.sort(key=lambda t: t[0])

        used_detections: set[int] = set()
        used_tracks: set[int] = set()
        for _, i, tid in candidate_pairs:
            if i in used_detections or tid in used_tracks:
                continue
            assigned[i] = tid
            used_detections.add(i)
            used_tracks.add(tid)
            track = self._tracks[tid]
            track.centroid = centroids[i]
            track.history.append((centroids[i], frame_index))
            track.frames_since_seen = 0

        # New tracks for unmatched detections.
        for i, c in enumerate(centroids):
            if assigned[i] is None:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = _Track(
                    track_id=tid, centroid=c, history=[(c, frame_index)]
                )
                assigned[i] = tid

        # Age and drop stale tracks.
        for tid in list(self._tracks.keys()):
            if tid not in used_tracks and tid not in [a for a in assigned]:
                self._tracks[tid].frames_since_seen += 1
                if self._tracks[tid].frames_since_seen > self.max_missed_frames:
                    del self._tracks[tid]

        return [a for a in assigned]  # type: ignore[return-value]

    def speed_px_per_frame(self, track_id: int, window: int = 5) -> float | None:
        """Average displacement per frame over the last ``window`` points.

        Returns ``None`` if the track doesn't have enough history yet.
        Multiply by frame rate (fps) to get px/second.
        """
        track = self._tracks.get(track_id)
        if track is None or len(track.history) < 2:
            return None
        points = track.history[-window:]
        if len(points) < 2:
            return None
        total_dist = 0.0
        total_frames = 0
        for (p1, f1), (p2, f2) in zip(points, points[1:]):
            total_dist += _euclidean(p1, p2)
            total_frames += max(1, f2 - f1)
        if total_frames == 0:
            return None
        return total_dist / total_frames


def _euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
