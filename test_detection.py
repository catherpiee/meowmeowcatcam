"""Unit tests for the pure hand/gesture classification logic in detection.py.

These build synthetic 21-point hand landmark sets (no webcam, no mediapipe)
and assert the classifier reads them the way a human would describe the pose.

Runnable two ways:
    pytest test_detection.py
    python test_detection.py        # falls back to a tiny built-in runner
"""

from collections import namedtuple

import detection as d

Landmark = namedtuple("Landmark", "x y z")

# MediaPipe hand landmark indices
WRIST = 0
THUMB = (1, 2, 3, 4)          # CMC, MCP, IP, TIP
INDEX = (5, 6, 7, 8)          # MCP, PIP, DIP, TIP
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)

# base x position + MCP y for each of the four fingers
_FINGER_BASE = {
    INDEX: (0.44, 0.62),
    MIDDLE: (0.50, 0.61),
    RING: (0.56, 0.62),
    PINKY: (0.62, 0.64),
}


def _finger_points(x, y_mcp, extended):
    """Return (mcp, pip, dip, tip) as (x, y) tuples for one finger."""
    if extended:
        # straight, pointing up (image y decreases upward); tip far from wrist
        return [(x, y_mcp), (x, y_mcp - 0.08), (x, y_mcp - 0.13), (x, y_mcp - 0.17)]
    # curled: folds back down so the tip sits near/below the MCP (closer to wrist)
    return [(x, y_mcp), (x, y_mcp - 0.05), (x - 0.005, y_mcp - 0.02), (x - 0.01, y_mcp + 0.02)]


def _thumb_points(extended):
    if extended:  # spread out to the side, straight
        return [(0.42, 0.78), (0.37, 0.72), (0.33, 0.67), (0.30, 0.63)]
    # tucked across the palm toward the index side
    return [(0.42, 0.78), (0.44, 0.72), (0.47, 0.68), (0.50, 0.65)]


def make_hand(index=True, middle=True, ring=True, pinky=True, thumb=True, z=0.0):
    """Build 21 Landmarks. Each finger flag True == extended, False == curled."""
    pts = [None] * 21
    pts[WRIST] = Landmark(0.50, 0.90, z)

    thumb_pts = _thumb_points(thumb)
    for landmark_idx, (px, py) in zip(THUMB, thumb_pts):
        pts[landmark_idx] = Landmark(px, py, z)

    flags = {INDEX: index, MIDDLE: middle, RING: ring, PINKY: pinky}
    for finger, (bx, by) in _FINGER_BASE.items():
        for landmark_idx, (px, py) in zip(finger, _finger_points(bx, by, flags[finger])):
            pts[landmark_idx] = Landmark(px, py, z)

    return pts


# ---------------------------------------------------------------------------
# finger_extended — directionality (item #2)
# ---------------------------------------------------------------------------
def test_straight_up_finger_is_extended():
    pts = [d.p3(lm) for lm in make_hand(index=True)]
    assert d.finger_extended(pts, INDEX[0], INDEX[1], INDEX[3]) is True


def test_curled_finger_is_not_extended():
    pts = [d.p3(lm) for lm in make_hand(index=False)]
    assert d.finger_extended(pts, INDEX[0], INDEX[1], INDEX[3]) is False


def test_straight_but_folded_back_finger_is_not_extended():
    # a finger that is geometrically straight but points back toward the wrist
    # (tip closer to the wrist than the PIP joint) must NOT read as extended.
    pts = [d.p3(lm) for lm in make_hand()]
    mcp, pip, tip = INDEX[0], INDEX[1], INDEX[3]
    wrist = pts[WRIST]
    # straighten the finger but aim it downward toward the wrist
    pts[mcp] = d.p3(Landmark(0.44, 0.55, 0.0))
    pts[pip] = d.p3(Landmark(0.46, 0.62, 0.0))
    pts[tip] = d.p3(Landmark(0.49, 0.74, 0.0))  # near the wrist
    assert d.finger_extended(pts, mcp, pip, tip) is False


# ---------------------------------------------------------------------------
# thumb_extended (item #3)
# ---------------------------------------------------------------------------
def test_thumb_out_is_extended():
    pts = [d.p3(lm) for lm in make_hand(thumb=True)]
    assert d.thumb_extended(pts) is True


def test_thumb_tucked_is_not_extended():
    pts = [d.p3(lm) for lm in make_hand(thumb=False)]
    assert d.thumb_extended(pts) is False


# ---------------------------------------------------------------------------
# classify_hand poses (items #2, #3)
# ---------------------------------------------------------------------------
def test_open_palm_all_five_extended():
    h = d.classify_hand(make_hand(True, True, True, True, thumb=True))
    assert h["fingersExtended"] == 4
    assert h["curledCount"] == 0
    assert h["thumbExtended"] is True


def test_fist_all_curled():
    h = d.classify_hand(make_hand(False, False, False, False, thumb=False))
    assert h["curledCount"] == 4
    assert h["thumbExtended"] is False


def test_pointing_index_only():
    h = d.classify_hand(make_hand(True, False, False, False, thumb=False))
    assert d.is_pointing(h) is True


def test_shaka_thumb_and_pinky_only():
    h = d.classify_hand(make_hand(index=False, middle=False, ring=False, pinky=True, thumb=True))
    assert h["pinkyUp"] is True
    assert h["thumbExtended"] is True
    assert h["indexUp"] is False and h["middleUp"] is False and h["ringUp"] is False


# ---------------------------------------------------------------------------
# dist2d ignores z (item #1)
# ---------------------------------------------------------------------------
def test_dist2d_ignores_z():
    a = d.p3(Landmark(0.0, 0.0, 0.0))
    b = d.p3(Landmark(3.0, 4.0, 99.0))
    assert abs(d.dist2d(a, b) - 5.0) < 1e-9


# ---------------------------------------------------------------------------
# palm normal / facing (item #4) — structural properties only.
# The camera sign is not asserted here (it can't be ground-truthed headlessly);
# it's surfaced in the HUD for on-camera tuning.
# ---------------------------------------------------------------------------
def test_palm_normal_is_unit_length():
    import numpy as np
    pts = [d.p3(lm) for lm in make_hand()]
    assert abs(float(np.linalg.norm(d.palm_normal(pts))) - 1.0) < 1e-6


def test_palm_facing_flips_with_handedness():
    pts = [d.p3(lm) for lm in make_hand()]
    assert d.palm_facing_camera(pts, "Left") != d.palm_facing_camera(pts, "Right")


# ---------------------------------------------------------------------------
# GestureState.decide routing + spin gating (items #1, #3, #5, #6)
# ---------------------------------------------------------------------------
import time  # noqa: E402
import numpy as np  # noqa: E402

from gesture_meme import GestureState  # noqa: E402

FakeHandResult = namedtuple("FakeHandResult", "hand_landmarks handedness")
FakeCategory = namedtuple("FakeCategory", "category_name score")


def _hand_result(hands, labels=None):
    """hands: list of landmark lists. labels: list of "Left"/"Right"/None."""
    if labels is None:
        labels = ["Right"] * len(hands)
    handedness = [[FakeCategory(lbl, 0.99)] if lbl else None for lbl in labels]
    return FakeHandResult(hand_landmarks=hands, handedness=handedness)


def _no_hands():
    return FakeHandResult(hand_landmarks=[], handedness=[])


def _set_face(state, yaw=0.0, mouth=(0.5, 0.5), face_width=0.2, seen=True, fresh=True):
    now = time.time() * 1000
    t = now if fresh else now - 10_000_000
    state.last_face = (np.array([mouth[0], mouth[1], 0.0]), face_width, 0.02, yaw, t)
    state.face_seen_this_frame = seen


def _set_spinning(state):
    now = time.time() * 1000
    thr = state.tun.spin_mag_threshold + 1.0
    state.flow_history = [(now - i * 30, thr) for i in range(40)]


def shift_hand(landmarks, dx=0.0, dy=0.0):
    return [Landmark(lm.x + dx, lm.y + dy, lm.z) for lm in landmarks]


def test_no_hands_neutral_head_is_default():
    s = GestureState()
    _set_face(s, yaw=3.0)
    assert s.decide(_no_hands()) == "default"


def test_no_hands_turned_head_is_side_eye():
    s = GestureState()
    _set_face(s, yaw=25.0)
    assert s.decide(_no_hands()) == "sideEyeCat"


def test_fist_is_detected():
    s = GestureState()
    _set_face(s, mouth=(0.5, 0.5))
    fist = make_hand(False, False, False, False, thumb=False)
    assert s.decide(_hand_result([fist])) == "fist"


def test_shaka_is_detected():
    s = GestureState()
    _set_face(s, mouth=(0.2, 0.2))  # face away from the hand
    shaka = make_hand(index=False, middle=False, ring=False, pinky=True, thumb=True)
    assert s.decide(_hand_result([shaka])) == "rockstar"


def test_index_near_mouth_is_shhh():
    s = GestureState()
    point = make_hand(True, False, False, False, thumb=False)
    tip = point[INDEX[3]]
    _set_face(s, mouth=(tip.x, tip.y), face_width=0.3)  # mouth right at the fingertip
    assert s.decide(_hand_result([point])) == "shhh"


def test_index_away_from_mouth_is_one_finger_up():
    s = GestureState()
    point = make_hand(True, False, False, False, thumb=False)
    _set_face(s, mouth=(0.9, 0.9), face_width=0.15)
    assert s.decide(_hand_result([point])) == "oneFingerUp"


def test_spin_does_not_override_a_hand_gesture():
    # item #6: a fast hand-wave that reads as a specific gesture must beat spin
    s = GestureState()
    _set_face(s, mouth=(0.5, 0.5))
    _set_spinning(s)
    fist = make_hand(False, False, False, False, thumb=False)
    assert s.decide(_hand_result([fist])) == "fist"


def test_spin_wins_when_no_specific_hand_gesture():
    s = GestureState()
    _set_face(s, fresh=False)
    _set_spinning(s)
    assert s.decide(_no_hands()) == "spinCat"


def test_two_pointing_fingers_together():
    s = GestureState()
    _set_face(s, mouth=(0.9, 0.1))
    left = make_hand(True, False, False, False, thumb=False)
    right = shift_hand(left, dx=0.05)  # index tips ~0.05 apart
    assert s.decide(_hand_result([left, right], labels=["Left", "Right"])) == "twoFingersTogether"


def test_two_hands_on_head():
    s = GestureState()
    _set_face(s, mouth=(0.5, 0.5), face_width=0.2)
    up = shift_hand(make_hand(), dy=-0.4)  # palms well above the head-top line
    assert s.decide(_hand_result([up, shift_hand(up, dx=0.1)],
                                 labels=["Left", "Right"])) == "twoHandsOnHead"


def test_two_hands_beside_face_is_crashout():
    s = GestureState()
    _set_face(s, mouth=(0.5, 0.5), face_width=0.2)
    beside = shift_hand(make_hand(), dy=-0.15)  # near the face but not above the head
    assert s.decide(_hand_result([beside, shift_hand(beside, dx=0.1)],
                                 labels=["Left", "Right"])) == "crashOutCat"


# ---------------------------------------------------------------------------
# stable_gesture — anti-flicker majority vote (item #8)
# ---------------------------------------------------------------------------
def test_single_stray_frame_does_not_flip_the_gesture():
    recent = ["fist", "fist", "fist", "fist", "crashOutCat"]
    assert d.stable_gesture(recent, "fist") == "fist"


def test_new_gesture_wins_once_it_holds_the_majority():
    recent = ["oneFingerUp", "oneFingerUp", "oneFingerUp", "fist", "fist"]
    assert d.stable_gesture(recent, "fist") == "oneFingerUp"


def test_no_majority_keeps_current():
    recent = ["a", "a", "b", "b", "c"]
    assert d.stable_gesture(recent, "b") == "b"


def test_empty_history_keeps_current():
    assert d.stable_gesture([], "default") == "default"


# ---------------------------------------------------------------------------
# tiny runner so the suite works without pytest installed
# ---------------------------------------------------------------------------
def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {fn.__name__}: {e or 'assertion failed'}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failures}/{len(fns)} passed")
    return failures


if __name__ == "__main__":
    import sys
    sys.exit(1 if _run_all() else 0)
