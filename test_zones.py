import pytest

from traffic_density.zones import Zone, ZoneManager, point_in_polygon

SQUARE = [(0, 0), (100, 0), (100, 100), (0, 100)]


def test_point_in_polygon_inside():
    assert point_in_polygon((50, 50), SQUARE) is True


def test_point_in_polygon_outside():
    assert point_in_polygon((150, 50), SQUARE) is False


def test_point_in_polygon_edge_cases_do_not_crash():
    # Points exactly on vertices / edges shouldn't raise, even if the
    # even-odd rule's boundary behaviour is implementation-defined.
    for pt in [(0, 0), (100, 0), (50, 0)]:
        point_in_polygon(pt, SQUARE)  # just must not raise


def test_zone_area_of_unit_square_scaled():
    zone = Zone(name="test", polygon=SQUARE, capacity=10)
    assert zone.area == pytest.approx(10000.0)


def test_zone_rejects_degenerate_polygon():
    with pytest.raises(ValueError):
        Zone(name="bad", polygon=[(0, 0), (1, 1)])


def test_zone_capacity_estimated_from_area_when_not_given():
    zone = Zone(name="auto", polygon=SQUARE, avg_vehicle_area_px=1000.0)
    # area 10000 / 1000 per vehicle = 10
    assert zone.capacity == 10


def test_zone_manager_rejects_duplicate_names():
    manager = ZoneManager()
    manager.add(Zone(name="lane_a", polygon=SQUARE))
    with pytest.raises(ValueError):
        manager.add(Zone(name="lane_a", polygon=SQUARE))


def test_zone_manager_assigns_point_to_correct_zone():
    manager = ZoneManager()
    left = Zone(name="left", polygon=[(0, 0), (50, 0), (50, 100), (0, 100)])
    right = Zone(name="right", polygon=[(50, 0), (100, 0), (100, 100), (50, 100)])
    manager.add(left)
    manager.add(right)

    assert manager.zone_for_point((10, 10)).name == "left"
    assert manager.zone_for_point((90, 10)).name == "right"
    assert manager.zone_for_point((500, 500)) is None


def test_zone_manager_from_config():
    config = {
        "zones": [
            {
                "name": "lane_north",
                "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
                "capacity": 5,
            }
        ]
    }
    manager = ZoneManager.from_config(config)
    assert manager.get("lane_north").capacity == 5


def test_zone_manager_get_missing_raises_keyerror():
    manager = ZoneManager()
    with pytest.raises(KeyError):
        manager.get("does_not_exist")
