import pytest

from traffic_density.density import (
    CongestionLevel,
    classify_congestion,
    compute_density,
)


@pytest.mark.parametrize(
    "count, capacity, expected",
    [
        (0, 10, 0.0),
        (5, 10, 0.5),
        (10, 10, 1.0),
        (15, 10, 1.0),  # clamped
        (0, 0, 0.0),  # zero capacity handled without ZeroDivisionError
    ],
)
def test_compute_density(count, capacity, expected):
    assert compute_density(count, capacity) == pytest.approx(expected)


@pytest.mark.parametrize(
    "density, expected",
    [
        (0.0, CongestionLevel.FREE_FLOW),
        (0.20, CongestionLevel.FREE_FLOW),
        (0.21, CongestionLevel.LIGHT),
        (0.45, CongestionLevel.LIGHT),
        (0.46, CongestionLevel.MODERATE),
        (0.70, CongestionLevel.MODERATE),
        (0.71, CongestionLevel.HEAVY),
        (0.90, CongestionLevel.HEAVY),
        (0.91, CongestionLevel.GRIDLOCK),
        (1.0, CongestionLevel.GRIDLOCK),
    ],
)
def test_classify_congestion_by_density_only(density, expected):
    assert classify_congestion(density) == expected


def test_slow_speed_escalates_congestion_by_one_level():
    # A low-density zone that would normally be FREE_FLOW gets bumped
    # to LIGHT if vehicles are barely moving.
    assert classify_congestion(0.1, avg_speed_px_s=5.0) == CongestionLevel.LIGHT


def test_slow_speed_escalation_caps_at_gridlock():
    assert classify_congestion(0.95, avg_speed_px_s=1.0) == CongestionLevel.GRIDLOCK


def test_fast_speed_does_not_escalate():
    assert classify_congestion(0.1, avg_speed_px_s=200.0) == CongestionLevel.FREE_FLOW
