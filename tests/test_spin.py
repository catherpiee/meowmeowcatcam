from gesture_meme import (
    SPIN_FRACTION_REQUIRED,
    SPIN_FRACTION_WINDOW_MS,
    SPIN_MAG_THRESHOLD,
    GestureState,
)

T0 = 1_000_000.0


def feed(state, samples, start=T0, step=50):
    """Push (magnitude) samples in at a fixed cadence. Returns the last timestamp."""
    t = start
    for mag in samples:
        state.update_flow(mag, 0.0, now=t)
        t += step
    return t - step


def test_no_samples_is_not_spinning():
    state = GestureState()
    assert state.is_spinning(T0) is False


def test_sustained_high_motion_is_spinning():
    state = GestureState()
    last = feed(state, [SPIN_MAG_THRESHOLD + 0.5] * 40)
    assert state.is_spinning(last) is True


def test_sustained_low_motion_is_not_spinning():
    state = GestureState()
    last = feed(state, [SPIN_MAG_THRESHOLD - 0.5] * 40)
    assert state.is_spinning(last) is False


def test_bursty_but_mostly_elevated_motion_still_counts_as_spinning():
    # a real spin slows and speeds back up; 3-of-4 frames elevated is ~0.75,
    # comfortably over the 0.55 fraction requirement
    state = GestureState()
    pattern = [SPIN_MAG_THRESHOLD + 0.5] * 3 + [SPIN_MAG_THRESHOLD - 0.5]
    last = feed(state, pattern * 10)
    assert state.is_spinning(last) is True


def test_a_single_burst_does_not_fill_the_window():
    # one ~0.5s burst inside a 2.2s window is well under the 0.55 fraction
    state = GestureState()
    burst = [SPIN_MAG_THRESHOLD + 0.5] * 10
    quiet = [SPIN_MAG_THRESHOLD - 0.5] * 34
    last = feed(state, burst + quiet)
    assert state.is_spinning(last) is False


def test_old_samples_age_out_of_the_window():
    state = GestureState()
    feed(state, [SPIN_MAG_THRESHOLD + 0.5] * 40)
    # jump past the whole trailing window: nothing is left to count
    assert state.is_spinning(T0 + SPIN_FRACTION_WINDOW_MS * 5) is False


def test_fraction_just_under_the_requirement_is_not_spinning():
    # window_samples is how many samples of a 50ms-cadence series survive the
    # trailing SPIN_FRACTION_WINDOW_MS window; elevated_needed is the largest
    # elevated count whose fraction is still <= SPIN_FRACTION_REQUIRED, i.e.
    # the boundary from below.
    step = 50
    window_samples = SPIN_FRACTION_WINDOW_MS // step
    elevated_needed = int(SPIN_FRACTION_REQUIRED * window_samples)
    pattern = [SPIN_MAG_THRESHOLD + 0.5] * elevated_needed + [
        SPIN_MAG_THRESHOLD - 0.5
    ] * (window_samples - elevated_needed)
    state = GestureState()
    last = feed(state, pattern, step=step)
    assert state.is_spinning(last) is False


def test_fraction_just_over_the_requirement_is_spinning():
    # one more elevated sample than the "just under" case above, pushing the
    # fraction just past SPIN_FRACTION_REQUIRED.
    step = 50
    window_samples = SPIN_FRACTION_WINDOW_MS // step
    elevated_needed = int(SPIN_FRACTION_REQUIRED * window_samples) + 1
    pattern = [SPIN_MAG_THRESHOLD + 0.5] * elevated_needed + [
        SPIN_MAG_THRESHOLD - 0.5
    ] * (window_samples - elevated_needed)
    state = GestureState()
    last = feed(state, pattern, step=step)
    assert state.is_spinning(last) is True
