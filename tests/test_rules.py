import math

from gesture_meme import HAND_COVER_FACE_DIST_FACE_SEEN, GestureState
from tests.builders import (
    face_result,
    hand_result,
    make_face,
    make_hand,
    move_hand,
    yaw_matrix,
)

T0 = 1_000_000.0
FACE_W = 0.2
MOUTH = (0.5, 0.5)


def state_with_face(*, yaw=0.0, mouth_open=0.02, now=T0):
    """A GestureState that has just seen a face at MOUTH with width FACE_W."""
    s = GestureState()
    s.update_face(
        face_result(make_face(center=MOUTH, width=FACE_W, mouth_open=mouth_open), yaw_matrix(yaw)),
        now=now,
    )
    return s


def at(hand, x, y):
    """Place a hand so its palmCenter (landmark 9) lands on (x, y)."""
    return move_hand(hand, x - hand[9].x, y - hand[9].y)


def fist():
    return make_hand(index_up=False, middle_up=False, ring_up=False, pinky_up=False)


def open_palm():
    return make_hand()


def pointing():
    return make_hand(index_up=True, middle_up=False, ring_up=False, pinky_up=False)


def rockstar_hand():
    return make_hand(
        index_up=False, middle_up=False, ring_up=False, pinky_up=True, thumb_out=True
    )


# ---- single-hand shapes ---------------------------------------------------

def test_no_hands_and_forward_face_is_default():
    s = state_with_face()
    assert s.decide(hand_result([]), now=T0) == "default"


def test_no_hands_and_turned_head_is_side_eye():
    s = state_with_face(yaw=30.0)
    assert s.decide(hand_result([]), now=T0) == "sideEyeCat"


def test_yaw_just_under_the_threshold_is_not_side_eye():
    s = state_with_face(yaw=14.0)
    assert s.decide(hand_result([]), now=T0) == "default"


def test_fist_far_from_the_face():
    s = state_with_face()
    h = at(fist(), 0.9, 0.9)
    assert s.decide(hand_result([h]), now=T0) == "fist"


def test_rockstar_far_from_the_face():
    s = state_with_face()
    h = at(rockstar_hand(), 0.9, 0.9)
    assert s.decide(hand_result([h]), now=T0) == "rockstar"


def test_open_palm_far_from_the_face_is_hand_stretched_out():
    s = state_with_face()
    h = at(open_palm(), 0.9, 0.9)
    assert s.decide(hand_result([h]), now=T0) == "handStretchedOut"


def test_pointing_away_from_the_mouth_is_one_finger_up():
    s = state_with_face()
    h = at(pointing(), 0.9, 0.9)
    assert s.decide(hand_result([h]), now=T0) == "oneFingerUp"


def test_pointing_with_the_tip_on_the_mouth_is_shhh():
    s = state_with_face()
    h = pointing()
    # put the index TIP (landmark 8) right on the mouth
    h = move_hand(h, MOUTH[0] - h[8].x, MOUTH[1] - h[8].y)
    assert s.decide(hand_result([h]), now=T0) == "shhh"


def test_open_palm_over_the_mouth_is_hand_cover_face():
    s = state_with_face()
    # 0.5 face-widths from the mouth: inside the 0.7 face-seen threshold
    h = at(open_palm(), MOUTH[0] + 0.5 * FACE_W, MOUTH[1])
    assert s.decide(hand_result([h]), now=T0) == "handCoverFace"


def test_hand_cover_face_uses_the_wider_radius_when_the_face_is_lost():
    s = state_with_face()
    s.update_face(face_result(None), now=T0 + 10)   # face lost this frame
    # 1.0 face-widths out: too far for the 0.7 threshold, inside the 1.3 one
    h = at(open_palm(), MOUTH[0] + 1.0 * FACE_W, MOUTH[1])
    assert s.decide(hand_result([h]), now=T0 + 10) == "handCoverFace"


# ---- two-hand shapes ------------------------------------------------------

def test_two_hands_above_the_head_is_two_hands_on_head():
    s = state_with_face()
    # head top is mouth_y - 1.1 * face_width = 0.5 - 0.22 = 0.28
    a = at(open_palm(), 0.42, 0.20)
    b = at(open_palm(), 0.58, 0.20)
    assert s.decide(hand_result([a, b]), now=T0) == "twoHandsOnHead"


def test_two_hands_beside_the_face_is_crash_out():
    s = state_with_face()
    # near the face (< 2.2 face-widths) but below the head-top line
    a = at(open_palm(), 0.35, 0.50)
    b = at(open_palm(), 0.65, 0.50)
    assert s.decide(hand_result([a, b]), now=T0) == "crashOutCat"


def test_only_one_hand_above_the_head_is_crash_out_not_two_hands_on_head():
    s = state_with_face()
    a = at(open_palm(), 0.42, 0.20)   # above
    b = at(open_palm(), 0.65, 0.50)   # beside
    assert s.decide(hand_result([a, b]), now=T0) == "crashOutCat"


def test_two_index_fingers_touching_is_two_fingers_together():
    s = state_with_face()
    a = pointing()
    b = pointing()
    # index tips within 1.4 hand-scales of each other, well away from the face
    a = move_hand(a, 0.85 - a[8].x, 0.85 - a[8].y)
    b = move_hand(b, 0.87 - b[8].x, 0.85 - b[8].y)
    assert s.decide(hand_result([a, b]), now=T0) == "twoFingersTogether"


def test_two_index_fingers_far_apart_is_not_two_fingers_together():
    s = state_with_face()
    a = at(pointing(), 0.10, 0.90)
    b = at(pointing(), 0.90, 0.90)
    assert s.decide(hand_result([a, b]), now=T0) != "twoFingersTogether"


# ---- priority ordering ----------------------------------------------------

def test_spinning_beats_every_hand_gesture():
    s = state_with_face()
    for i in range(40):
        s.update_flow(5.0, 1.0, now=T0 + i * 50)
    h = at(fist(), 0.9, 0.9)
    assert s.decide(hand_result([h]), now=T0 + 39 * 50) == "spinCat"


def test_shhh_beats_hand_cover_face():
    # a shhh pose IS a hand near the face: the index-only branch (which can
    # return "shhh") must be reached before the broader hand-covering-face
    # branch, or a shhh pose gets swallowed into "handCoverFace" instead.
    # Prove the two branches actually disagree on this pose before trusting
    # the "shhh" result to mean anything about ordering: fingertip on the
    # mouth makes the shhh distance 0 (unambiguously shhh), and the same
    # pose's palm still lands inside the hand-cover-face radius (measured
    # from a different landmark), so if handCoverFace were checked first it
    # would fire instead.
    s = state_with_face()
    h = pointing()
    h = move_hand(h, MOUTH[0] - h[8].x, MOUTH[1] - h[8].y)

    palm_dist_face_widths = math.hypot(h[9].x - MOUTH[0], h[9].y - MOUTH[1]) / FACE_W
    assert palm_dist_face_widths < HAND_COVER_FACE_DIST_FACE_SEEN, (
        "this pose must also satisfy the hand-cover-face distance check, "
        "otherwise this test cannot distinguish correct ordering from broken ordering"
    )

    assert s.decide(hand_result([h]), now=T0) == "shhh"


def test_two_hands_on_head_beats_crash_out():
    s = state_with_face()
    a = at(open_palm(), 0.42, 0.15)
    b = at(open_palm(), 0.58, 0.15)
    assert s.decide(hand_result([a, b]), now=T0) == "twoHandsOnHead"


def test_a_stale_face_disables_every_face_dependent_rule():
    s = state_with_face(yaw=30.0)
    stale = T0 + 5_000
    # side-eye needs a fresh face
    assert s.decide(hand_result([]), now=stale) == "default"
    # so does hand-cover-face: an open palm at the mouth falls through to
    # handStretchedOut once the face memory has expired
    h = at(open_palm(), MOUTH[0], MOUTH[1])
    assert s.decide(hand_result([h]), now=stale) == "handStretchedOut"


def test_no_face_at_all_still_classifies_hand_shapes():
    s = GestureState()
    h = at(fist(), 0.9, 0.9)
    assert s.decide(hand_result([h]), now=T0) == "fist"


# ---- deterministic hand selection ----------------------------------------

def test_gesture_is_the_same_whichever_order_mediapipe_returns_the_hands():
    s = state_with_face()
    gesturing = at(fist(), 0.15, 0.90)     # a deliberate fist
    resting = at(open_palm(), 0.85, 0.90)  # the other hand, just present

    forward = s.decide(hand_result([gesturing, resting]), now=T0)
    backward = s.decide(hand_result([resting, gesturing]), now=T0)

    assert forward == backward


def test_a_deliberate_gesture_wins_over_a_resting_hand():
    s = state_with_face()
    gesturing = at(rockstar_hand(), 0.15, 0.90)
    resting = at(open_palm(), 0.85, 0.90)

    assert s.decide(hand_result([resting, gesturing]), now=T0) == "rockstar"


def test_hand_order_is_stable_across_repeated_calls():
    s = state_with_face()
    a = at(fist(), 0.15, 0.90)
    b = at(fist(), 0.85, 0.90)
    results = {s.decide(hand_result([a, b]), now=T0 + i) for i in range(10)}
    assert len(results) == 1


def test_gesture_is_stable_when_two_simultaneous_gestures_jitter_across_the_position_sort():
    # Two hands near-equal x, each forming a *different* valid gesture (a
    # position-only sort is prone to jitter flipping which one sorts first
    # frame to frame - this pins the same physical hand's gesture across
    # that jitter via the stable handedness label instead).
    s = state_with_face()
    # y kept far enough from the face (dist/face_width > 2.2) that the
    # two-hands-near-face rules don't intercept these before the ladder runs.
    fist_hand = at(fist(), 0.5000, 0.98)  # "Left" hand
    rockstar_a = at(rockstar_hand(), 0.5001, 0.98)  # "Right" hand, first frame: right of the fist
    rockstar_b = at(rockstar_hand(), 0.4999, 0.98)  # same "Right" hand, jittered left of the fist

    first = s.decide(hand_result([fist_hand, rockstar_a], handedness=["Left", "Right"]), now=T0)
    second = s.decide(hand_result([fist_hand, rockstar_b], handedness=["Left", "Right"]), now=T0 + 1)

    assert first == second


def test_every_gesture_has_at_least_one_meme_file():
    from pathlib import Path

    from gesture_meme import GESTURE_MEMES, MEMES

    for gesture, files in GESTURE_MEMES.items():
        assert files, f"{gesture} has no meme files"
        for name in files:
            assert (MEMES / name).exists(), f"missing meme file for {gesture}: {name}"
