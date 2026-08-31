from traffic_density.schemas import Detection, FrameReport, ZoneReport


def test_detection_centroid():
    d = Detection(x1=0, y1=0, x2=10, y2=20, class_name="car", confidence=0.9)
    assert d.centroid == (5.0, 10.0)


def test_detection_area():
    d = Detection(x1=0, y1=0, x2=10, y2=20, class_name="car", confidence=0.9)
    assert d.area == 200.0


def test_detection_area_never_negative_for_malformed_box():
    d = Detection(x1=10, y1=10, x2=0, y2=0, class_name="car", confidence=0.9)
    assert d.area == 0.0


def test_frame_report_as_dict_roundtrips_zone_data():
    zr = ZoneReport(
        zone_name="lane_a",
        frame_index=1,
        counts_by_class={"car": 2},
        total_count=2,
        density_score=0.2,
        congestion_level="LIGHT",
    )
    fr = FrameReport(frame_index=1, timestamp_s=0.033, zones=[zr])
    d = fr.as_dict()
    assert d["zones"][0]["zone_name"] == "lane_a"
    assert d["zones"][0]["total_count"] == 2
