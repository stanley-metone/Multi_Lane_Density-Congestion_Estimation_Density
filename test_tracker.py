from traffic_density.tracker import CentroidTracker


def test_new_detection_gets_new_track_id():
    tracker = CentroidTracker()
    ids = tracker.update([(10, 10)], frame_index=0)
    assert len(ids) == 1
    assert ids[0] == 0


def test_same_object_across_frames_keeps_same_id():
    tracker = CentroidTracker(max_distance_px=50)
    ids0 = tracker.update([(10, 10)], frame_index=0)
    ids1 = tracker.update([(15, 12)], frame_index=1)  # small displacement
    assert ids0[0] == ids1[0]


def test_far_away_point_gets_new_id_not_matched():
    tracker = CentroidTracker(max_distance_px=20)
    ids0 = tracker.update([(10, 10)], frame_index=0)
    ids1 = tracker.update([(500, 500)], frame_index=1)  # far away, new object
    assert ids0[0] != ids1[0]


def test_track_dropped_after_max_missed_frames():
    tracker = CentroidTracker(max_distance_px=50, max_missed_frames=2)
    tracker.update([(10, 10)], frame_index=0)
    tracker.update([], frame_index=1)
    tracker.update([], frame_index=2)
    tracker.update([], frame_index=3)  # 3rd consecutive miss -> dropped
    assert tracker.active_track_ids == []


def test_speed_returns_none_with_insufficient_history():
    tracker = CentroidTracker()
    ids = tracker.update([(0, 0)], frame_index=0)
    assert tracker.speed_px_per_frame(ids[0]) is None


def test_speed_reports_average_displacement_per_frame():
    tracker = CentroidTracker(max_distance_px=50)
    tid = tracker.update([(0, 0)], frame_index=0)[0]
    tracker.update([(10, 0)], frame_index=1)
    tracker.update([(20, 0)], frame_index=2)
    speed = tracker.speed_px_per_frame(tid)
    assert speed == 10.0  # moved 10px each frame


def test_multiple_objects_matched_independently():
    tracker = CentroidTracker(max_distance_px=30)
    ids0 = tracker.update([(0, 0), (100, 100)], frame_index=0)
    ids1 = tracker.update([(5, 5), (105, 105)], frame_index=1)
    assert ids0[0] == ids1[0]
    assert ids0[1] == ids1[1]
    assert ids0[0] != ids0[1]
