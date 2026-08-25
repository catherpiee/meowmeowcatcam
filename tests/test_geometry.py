import math

import numpy as np
import pytest

from gesture_meme import (
    angle_deg,
    classify_hand,
    dist,
    finger_extended,
    p3,
    yaw_from_transform_matrix,
)
from tests.builders import make_hand, yaw_matrix


def test_angle_deg_straight_and_right_angle():
    assert angle_deg([1, 0, 0], [1, 0, 0]) == pytest.approx(0.0)
    assert angle_deg([1, 0, 0], [0, 1, 0]) == pytest.approx(90.0)
    assert angle_deg([1, 0, 0], [-1, 0, 0]) == pytest.approx(180.0)


def test_angle_deg_zero_magnitude_returns_180():
    # a degenerate vector must not blow up; the code treats it as maximally bent
    assert angle_deg([0, 0, 0], [1, 0, 0]) == 180.0


def test_finger_extended_straddles_the_45_degree_boundary():
    # mcp -> pip points straight up; tip is bent by a known angle from there.
    # finger_extended subtracts these, so they must be numpy arrays - which is
    # what classify_hand feeds it in production (via p3).
    def pts_for(bend_deg):
        t = math.radians(bend_deg)
        return {
            0: np.array([0.0, 0.0, 0.0]),   # mcp
            1: np.array([0.0, -1.0, 0.0]),  # pip
            2: np.array([math.sin(t), -1.0 - math.cos(t), 0.0]),  # tip
        }

    assert finger_extended(pts_for(44.0), 0, 1, 2) is True
    assert finger_extended(pts_for(46.0), 0, 1, 2) is False


def test_classify_hand_reports_each_finger_state():
    h = classify_hand(make_hand(index_up=True, middle_up=False, ring_up=False, pinky_up=False))
    assert h["indexUp"] is True
    assert h["middleUp"] is False
    assert h["ringUp"] is False
    assert h["pinkyUp"] is False
    assert h["curledCount"] == 3


def test_classify_hand_counts_a_fist_as_four_curled():
    h = classify_hand(make_hand(index_up=False, middle_up=False, ring_up=False, pinky_up=False))
    assert h["curledCount"] == 4


def test_classify_hand_counts_an_open_palm_as_zero_curled():
    h = classify_hand(make_hand())
    assert h["curledCount"] == 0


def test_classify_hand_thumb_out_flag():
    assert classify_hand(make_hand(thumb_out=True))["thumbOut"] is True
    assert classify_hand(make_hand(thumb_out=False))["thumbOut"] is False


def test_classify_hand_scale_matches_the_builder_scale():
    h = classify_hand(make_hand(scale=0.1))
    assert h["handScale"] == pytest.approx(0.1)


def test_classify_hand_palm_center_is_landmark_9():
    lms = make_hand(center=(0.4, 0.6))
    h = classify_hand(lms)
    assert h["palmCenter"][0] == pytest.approx(0.4)
    assert h["palmCenter"][1] == pytest.approx(0.6)


@pytest.mark.parametrize("deg", [-60.0, -15.0, 0.0, 15.0, 60.0])
def test_yaw_from_transform_matrix_round_trips(deg):
    assert yaw_from_transform_matrix(yaw_matrix(deg)) == pytest.approx(deg, abs=1e-6)


def test_yaw_from_transform_matrix_returns_zero_at_the_gimbal_singularity():
    # a 90-degree yaw makes sy fall below 1e-6, and the code bails to 0.0
    # rather than reading a meaningless atan2. Locking this in so the guard
    # is not silently removed.
    m = [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0, 0, 0, 1]]
    assert yaw_from_transform_matrix(m) == 0.0
