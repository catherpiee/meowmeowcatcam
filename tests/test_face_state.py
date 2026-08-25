import pytest

from gesture_meme import FACE_STALE_MS, GestureState
from tests.builders import face_result, hand_result, make_face, yaw_matrix

T0 = 1_000_000.0


def test_update_face_records_metrics_at_the_given_time():
    state = GestureState()
    state.update_face(face_result(make_face(center=(0.5, 0.5), width=0.2), yaw_matrix(0)), now=T0)

    mouth_center, face_width, mouth_open, yaw_deg, t = state.last_face
    assert mouth_center[0] == 0.5
    assert face_width == pytest.approx(0.2)
    assert t == T0


def test_update_face_computes_mouth_open_as_a_ratio_of_face_width():
    state = GestureState()
    state.update_face(face_result(make_face(width=0.2, mouth_open=0.25)), now=T0)
    assert state.last_face[2] == pytest.approx(0.25)


def test_update_face_reads_yaw_from_the_transform_matrix():
    state = GestureState()
    state.update_face(face_result(make_face(), yaw_matrix(30.0)), now=T0)
    assert round(state.last_face[3]) == 30


def test_update_face_with_no_face_keeps_the_last_one_but_clears_seen_flag():
    state = GestureState()
    state.update_face(face_result(make_face()), now=T0)
    state.update_face(face_result(None), now=T0 + 10)

    assert state.last_face is not None      # memory is retained
    assert state.face_seen_this_frame is False


def test_a_fresh_face_still_drives_side_eye():
    state = GestureState()
    state.update_face(face_result(make_face(), yaw_matrix(30.0)), now=T0)
    # just inside the staleness window
    assert state.decide(hand_result([]), now=T0 + FACE_STALE_MS - 1) == "sideEyeCat"


def test_a_stale_face_stops_driving_side_eye():
    state = GestureState()
    state.update_face(face_result(make_face(), yaw_matrix(30.0)), now=T0)
    # one millisecond past the staleness window
    assert state.decide(hand_result([]), now=T0 + FACE_STALE_MS + 1) == "default"
