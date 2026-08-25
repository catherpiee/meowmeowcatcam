import random

import pytest

from gesture_meme import (
    DEFAULT_FALLBACK_MS,
    MIN_HOLD_MS,
    STABLE_FRAMES_REQUIRED,
    MemeSelector,
)

T0 = 1_000_000.0
MEMES = {"default": ["poker"], "fist": ["punch"], "rockstar": ["cat"]}


def selector():
    # `now` MUST be passed: MemeSelector defaults it to the wall clock, and a
    # wall-clock switched_at compared against a synthetic T0 makes every hold
    # window come out negative.
    return MemeSelector(MEMES, rng=random.Random(1234), now=T0)


def push(sel, gesture, count, start, step=50):
    """Feed the same gesture `count` times. Returns the last timestamp used."""
    t = start
    for _ in range(count):
        sel.update(gesture, t)
        t += step
    return t - step


def test_starts_on_default():
    sel = selector()
    assert sel.current_gesture == "default"
    assert sel.current_meme == "poker"


def test_a_streak_shorter_than_required_does_not_switch():
    sel = selector()
    push(sel, "fist", STABLE_FRAMES_REQUIRED - 1, T0)
    assert sel.current_gesture == "default"


def test_a_full_streak_switches():
    sel = selector()
    push(sel, "fist", STABLE_FRAMES_REQUIRED, T0)
    assert sel.current_gesture == "fist"
    assert sel.current_meme == "punch"


def test_alternating_classifications_never_switch():
    # A,B,A,B... must not accumulate a streak for either value
    sel = selector()
    t = T0
    for i in range(40):
        sel.update("fist" if i % 2 == 0 else "rockstar", t)
        t += 50
    assert sel.current_gesture == "default"


def test_a_new_gesture_cannot_replace_one_inside_the_hold_window():
    sel = selector()
    t = push(sel, "fist", STABLE_FRAMES_REQUIRED, T0)
    assert sel.current_gesture == "fist"

    # a full streak of a different gesture, entirely inside the hold window
    push(sel, "rockstar", STABLE_FRAMES_REQUIRED, t + 1, step=10)
    assert sel.current_gesture == "fist"


def test_alternating_non_default_classifications_fall_back_to_default():
    # Establish "fist" as displayed, past both STABLE_FRAMES_REQUIRED and
    # MIN_HOLD_MS, so the hold guard is not what keeps it displayed. Then
    # alternate two different non-default gestures for longer than
    # DEFAULT_FALLBACK_MS: neither ever forms a stable streak, so the display
    # must not stay frozen on the stale "fist" reading -- it should revert to
    # "default".
    sel = selector()
    t = push(sel, "fist", STABLE_FRAMES_REQUIRED, T0)
    assert sel.current_gesture == "fist"

    # keep classifying "fist" a while longer so we clear MIN_HOLD_MS
    t = push(sel, "fist", 10, t + 50, step=100)
    assert sel.current_gesture == "fist"

    # now alternate between two different non-default gestures for well
    # over DEFAULT_FALLBACK_MS without either ever repeating back to back
    t2 = t
    for i in range(30):
        t2 += 50
        sel.update("rockstar" if i % 2 == 0 else "fist", t2)
    assert t2 - t > DEFAULT_FALLBACK_MS

    assert sel.current_gesture == "default"


def test_a_new_gesture_replaces_the_old_one_after_the_hold_expires():
    sel = selector()
    t = push(sel, "fist", STABLE_FRAMES_REQUIRED, T0)
    later = t + MIN_HOLD_MS + 1
    push(sel, "rockstar", STABLE_FRAMES_REQUIRED, later)
    assert sel.current_gesture == "rockstar"


def test_falls_back_to_default_after_the_fallback_window():
    sel = selector()
    t = push(sel, "fist", STABLE_FRAMES_REQUIRED, T0)
    # nothing but default from here on, past both the hold and the fallback
    push(sel, "default", 20, t + MIN_HOLD_MS + 1, step=100)
    assert sel.current_gesture == "default"


def test_does_not_fall_back_to_default_too_early():
    sel = selector()
    t = push(sel, "fist", STABLE_FRAMES_REQUIRED, T0)
    sel.update("default", t + DEFAULT_FALLBACK_MS - 100)
    assert sel.current_gesture == "fist"


def test_the_meme_image_is_stable_while_a_gesture_is_held():
    sel = MemeSelector(
        {"default": ["poker"], "fist": ["a", "b", "c", "d", "e"]},
        rng=random.Random(7),
        now=T0,
    )
    t = push(sel, "fist", STABLE_FRAMES_REQUIRED, T0)
    chosen = sel.current_meme
    push(sel, "fist", 30, t + 50)
    assert sel.current_meme == chosen


def test_min_hold_does_not_block_the_default_fallback():
    # MIN_HOLD_MS (700) exceeds DEFAULT_FALLBACK_MS (600). Prove the hold
    # does not block the fallback: at 650ms since the switch to "fist" --
    # past DEFAULT_FALLBACK_MS but short of MIN_HOLD_MS -- a single default
    # reading must still revert the display to "default".
    sel = selector()
    t = push(sel, "fist", STABLE_FRAMES_REQUIRED, T0)
    sel.update("default", t + 650)
    assert sel.current_gesture == "default"
