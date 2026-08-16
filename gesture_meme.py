"""
Webcam gesture -> meme detector (desktop version).

Opens two windows, side by side like the OBS/streamer setups:
  - "Camera": your webcam feed with hand landmarks drawn on top
  - "Meme": the cat meme matching whatever gesture you're making

Gestures:
  rockstar / shaka  -> memes/cat.jpg
  default (no hand) -> memes/pokercat.jpg
  one finger up     -> memes/profcat.jpg, memes/professorcat.jpg
  fist / punch      -> memes/punchcat.jpg
  shhh              -> memes/shhcat.jpg
  two fingers together (both hands, tips touching) -> memes/uwucat.jpg, memes/uwucatt.jpg,
                                                        memes/fingers together muehehe .jpg
  hand covering face -> memes/hand cover face .jpg
  crash-out cat (two hands up beside the face)            -> memes/crashout cat .jpg
  two hands on head                                        -> memes/two hands on head .jpg
  hand stretched out, palm facing camera (open hand)       -> memes/hand stretched out, palm facing up .jpg
  side eye (head turned to the side)                       -> memes/side eye cat.jpg
  spin cat (spinning fast in your chair)                   -> memes/spin cat.mov (plays as a video)
  screaming (mouth wide open)                              -> memes/screaming cat.jpg
  huh? (head tilted sideways)                              -> memes/huh cat.jpg

A gesture whose meme file isn't in memes/ yet is reported at startup and
simply never fires, so new gestures can be wired up before their artwork
exists.

The Camera window shows a compact live readout in the top-left corner -
each line is "value/threshold" for the signals that drive a gesture (head
yaw, head roll, mouth openness, and optical-flow magnitude/fraction), so
they can all be tuned by eye. See SIDE_EYE_YAW_DEG, HUH_ROLL_DEG,
SCREAM_MOUTH_OPEN and the SPIN_* constants below.

Press q or ESC to quit.
"""

import math
import random
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
from mediapipe import Image, ImageFormat

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
MEMES = ROOT / "memes"

GESTURE_MEMES = {
    "rockstar": ["cat.jpg"],
    "default": ["pokercat.jpg"],
    "oneFingerUp": ["profcat.jpg", "professorcat.jpg"],
    "fist": ["punchcat.jpg"],
    "shhh": ["shhcat.jpg"],
    "twoFingersTogether": ["uwucat.jpg", "uwucatt.jpg", "fingers together muehehe .jpg"],
    "handCoverFace": ["hand cover face .jpg"],
    "crashOutCat": ["crashout cat .jpg"],
    "twoHandsOnHead": ["two hands on head .jpg"],
    "handStretchedOut": ["hand stretched out, palm facing up .jpg"],
    "sideEyeCat": ["side eye cat.jpg"],
    "spinCat": ["spin cat.mov"],
    "screamingCat": ["screaming cat.jpg"],
    "huhCat": ["huh cat.jpg"],
}

# gestures whose meme is a video, not a still image
VIDEO_GESTURES = {"spinCat"}

STABLE_FRAMES_REQUIRED = 5
DEFAULT_FALLBACK_MS = 600
FACE_STALE_MS = 1200

# how far the head has to turn (yaw, in degrees, from MediaPipe's own head
# pose estimate - not a hand-rolled distance heuristic) to count as a
# side-eye look. Watch the live "yaw" readout in the Camera window while
# turning your head to find the right value for you.
SIDE_EYE_YAW_DEG = 15.0

# how far the head has to tilt sideways (roll, degrees, same head-pose
# estimate) for the "huh?" head-tilt look. Set well above SIDE_EYE_YAW_DEG's
# neighbourhood of casual movement - a deliberate tilt is a big, obvious
# motion, and a small roll is just how people naturally hold their head.
# Watch the live "roll" readout while tilting to tune it.
HUH_ROLL_DEG = 18.0

# how far open the mouth has to be to count as a scream. Measured as
# lip-gap over face height (chin to forehead), so it changes neither with
# distance from the camera nor with the capture aspect ratio: a closed
# mouth sits near 0, talking peaks around 0.08, and a deliberate
# wide-open scream runs past 0.15. Watch the live "mouth" readout in the
# Camera window to tune it for your face.
SCREAM_MOUTH_OPEN = 0.15

# spin detection: full-frame optical flow, downsized for speed. We compute
# magnitude (how much of the frame moved, on average) each frame; coherence
# (what fraction of that motion agreed on one direction) is also computed
# and logged for reference, but real recorded data showed it wasn't adding
# discrimination - averaging magnitude across the whole frame already dilutes
# out small localized motions (a hand gesture only fills a fraction of the
# frame, so the frame-wide average stays low regardless of coherence).
#
# What actually separates a real spin from a quick lean/reach turned out to
# be less about "how high does it peak" (both can peak similarly for an
# instant) and more about *how much of a multi-second window stays elevated*.
# A real spin is naturally bursty - you slow down, reposition, speed back
# up - so requiring one perfectly unbroken stretch above threshold was too
# strict and rejected real spins. Instead: over a trailing ~2.2s window,
# what fraction of frames had magnitude above a modest threshold? A real
# spin (even a "weak"/bursty one) kept that fraction above ~0.9; a one-off
# lean/reach can only fill a fraction of a multi-second window before it
# settles back down.
#
# Tuned from two real recorded sessions (flow_debug_log.csv, regenerated
# each run):
#   real spin (strong)  -> fraction above 0.8 stayed near 0.9-1.0
#   real spin (weaker)  -> fraction above 0.8 peaked at 0.92-0.93
#   fast sideways lean   -> a single ~1s burst, well under half of any 2s+ window
# If it's still misfiring or not firing for you, flow_debug_log.csv has the
# raw numbers from your most recent run - report back what fraction your
# non-spin motions vs your spins actually reach so this can be re-tuned to
# your setup.
SPIN_FLOW_WIDTH = 160
SPIN_FLOW_HEIGHT = 90
SPIN_FLOW_NOISE_FLOOR_PX = 0.4  # per-pixel motion below this is treated as noise, not real motion
SPIN_FLOW_MIN_MOVING_FRACTION = 0.15  # need at least this much of the frame moving to trust coherence at all
SPIN_MAG_THRESHOLD = 0.8  # per-frame magnitude counted as "elevated" for the fraction test
SPIN_FRACTION_WINDOW_MS = 2200  # trailing window the fraction is measured over
SPIN_FRACTION_REQUIRED = 0.55  # fraction of that window that must be elevated to count as spinning
SPIN_FLOW_PEAK_HOLD_MS = 2000

# hand-covering-face: how close the hand needs to be to where the mouth
# last was. Wider when the face detector has fully lost the face (strong
# evidence of a real occlusion); tighter when the face is still partially
# tracked (weaker evidence, avoid false positives from a hand just passing
# near the face).
HAND_COVER_FACE_DIST_FACE_LOST = 1.3
HAND_COVER_FACE_DIST_FACE_SEEN = 0.7

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# ---- geometry helpers (ported from the JS version) -----------------------
def p3(lm):
    return np.array([lm.x, lm.y, lm.z])


def dist(a, b):
    return float(np.linalg.norm(a - b))


def angle_deg(v1, v2):
    m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if m1 < 1e-9 or m2 < 1e-9:
        return 180.0
    cos_a = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def finger_extended(pts, mcp, pip, tip):
    v1 = pts[pip] - pts[mcp]
    v2 = pts[tip] - pts[pip]
    return angle_deg(v1, v2) < 45


def yaw_roll_from_transform_matrix(matrix):
    """Extract the head's left/right turn (yaw) and sideways tilt (roll),
    in degrees, from MediaPipe's facial transformation matrix - its own
    estimate of head pose, far more robust than trying to infer either
    from landmark distances.

    Yaw drives side-eye; roll drives the head-tilt "huh?" look.
    """
    r = np.asarray(matrix)[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy < 1e-6:
        return 0.0, 0.0
    yaw = math.atan2(-r[2, 0], sy)
    roll = math.atan2(r[1, 0], r[0, 0])
    return math.degrees(yaw), math.degrees(roll)


def classify_hand(landmarks):
    pts = [p3(lm) for lm in landmarks]
    hand_scale = dist(pts[0], pts[9]) or 1e-6

    index_up = finger_extended(pts, 5, 6, 8)
    middle_up = finger_extended(pts, 9, 10, 12)
    ring_up = finger_extended(pts, 13, 14, 16)
    pinky_up = finger_extended(pts, 17, 18, 20)

    thumb_pinky_spread = dist(pts[4], pts[17]) / hand_scale
    thumb_out = thumb_pinky_spread > 1.05

    curled_count = sum(1 for v in (index_up, middle_up, ring_up, pinky_up) if not v)

    return {
        "indexUp": index_up,
        "middleUp": middle_up,
        "ringUp": ring_up,
        "pinkyUp": pinky_up,
        "thumbOut": thumb_out,
        "curledCount": curled_count,
        "handScale": hand_scale,
        "indexTip": pts[8],
        "wrist": pts[0],
        "palmCenter": pts[9],
    }


def is_pointing(h):
    return h["indexUp"] and not h["middleUp"] and not h["ringUp"] and not h["pinkyUp"]


def frame_flow_signal(frame, prev_small_gray):
    """Downsize + compute dense optical flow against the previous frame,
    then reduce it to (magnitude, coherence): how much of the frame moved
    on the horizontal axis, and what fraction of that motion agreed on one
    direction. Returns (magnitude, coherence, small_gray_for_next_call)."""
    small = cv2.resize(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (SPIN_FLOW_WIDTH, SPIN_FLOW_HEIGHT)
    )
    if prev_small_gray is None:
        return 0.0, 0.0, small

    flow = cv2.calcOpticalFlowFarneback(
        prev_small_gray, small, None, 0.5, 2, 15, 2, 5, 1.2, 0
    )
    flow_x = flow[..., 0]

    magnitude = float(np.abs(flow_x).mean())

    moving_mask = np.abs(flow_x) > SPIN_FLOW_NOISE_FLOOR_PX
    moving_count = int(moving_mask.sum())
    total = flow_x.size
    if moving_count / total < SPIN_FLOW_MIN_MOVING_FRACTION:
        coherence = 0.0
    else:
        mean_sign = np.sign(flow_x[moving_mask].mean())
        if mean_sign == 0:
            coherence = 0.0
        else:
            agree = int((np.sign(flow_x[moving_mask]) == mean_sign).sum())
            coherence = agree / moving_count

    return magnitude, coherence, small


class GestureState:
    def __init__(self):
        self.last_face = None  # (mouth_center, face_width, mouth_open, yaw_deg, roll_deg, t)
        self.face_seen_this_frame = False
        self.last_yaw_debug = 0.0
        self.last_roll_debug = 0.0
        self.last_mouth_open_debug = 0.0
        self.flow_history = []  # [(t, magnitude), ...] trailing samples, for the fraction-above trigger
        self.flow_peak_history = []  # [(t, score), ...] longer trailing window, for the readable peak display
        self.last_flow_magnitude_debug = 0.0
        self.last_flow_coherence_debug = 0.0
        self.last_flow_score_debug = 0.0
        self.last_flow_peak_debug = 0.0
        self.last_flow_fraction_debug = 0.0

    def update_flow(self, magnitude, coherence):
        now = time.time() * 1000
        score = magnitude * coherence  # kept for the debug HUD/log only, not the trigger

        self.flow_history.append((now, magnitude))
        self.flow_history = [(t, m) for t, m in self.flow_history if now - t < SPIN_FRACTION_WINDOW_MS]

        self.flow_peak_history.append((now, score))
        self.flow_peak_history = [
            (t, s) for t, s in self.flow_peak_history if now - t < SPIN_FLOW_PEAK_HOLD_MS
        ]

        self.last_flow_magnitude_debug = magnitude
        self.last_flow_coherence_debug = coherence
        self.last_flow_score_debug = score
        self.last_flow_peak_debug = max((s for _, s in self.flow_peak_history), default=0.0)
        elevated = sum(1 for _, m in self.flow_history if m > SPIN_MAG_THRESHOLD)
        self.last_flow_fraction_debug = elevated / len(self.flow_history) if self.flow_history else 0.0

    def is_spinning(self, now):
        self.flow_history = [(t, m) for t, m in self.flow_history if now - t < SPIN_FRACTION_WINDOW_MS]
        if not self.flow_history:
            return False
        elevated = sum(1 for _, m in self.flow_history if m > SPIN_MAG_THRESHOLD)
        fraction = elevated / len(self.flow_history)
        return fraction > SPIN_FRACTION_REQUIRED

    def update_face(self, face_result):
        now = time.time() * 1000
        saw_face = bool(face_result.face_landmarks)

        if saw_face:
            f = face_result.face_landmarks[0]
            upper_lip, lower_lip = p3(f[13]), p3(f[14])
            right_cheek, left_cheek = p3(f[234]), p3(f[454])
            forehead, chin = p3(f[10]), p3(f[152])
            mouth_center = (upper_lip + lower_lip) / 2
            face_width = dist(right_cheek, left_cheek)
            # Normalize the lip gap against face HEIGHT, not width. Landmark
            # x and y are normalized against the frame's width and height
            # separately, so a vertical measurement over a horizontal one
            # silently scales with the capture aspect ratio - the same
            # scream would read differently at 16:9 than at 4:3. Chin to
            # forehead is vertical like the lip gap, so that cancels out.
            mouth_open = dist(upper_lip, lower_lip) / max(dist(forehead, chin), 1e-6)

            yaw_deg = roll_deg = 0.0
            if face_result.facial_transformation_matrixes:
                yaw_deg, roll_deg = yaw_roll_from_transform_matrix(
                    face_result.facial_transformation_matrixes[0]
                )

            self.last_face = (mouth_center, face_width, mouth_open, yaw_deg, roll_deg, now)
            self.last_yaw_debug = yaw_deg
            self.last_roll_debug = roll_deg
            self.last_mouth_open_debug = mouth_open
        self.face_seen_this_frame = saw_face

    def decide(self, hand_result):
        now = time.time() * 1000
        face_is_fresh = self.last_face is not None and now - self.last_face[5] < FACE_STALE_MS

        # spinning in the chair beats everything else, hands included.
        if self.is_spinning(now):
            return "spinCat"

        # a wide-open mouth is a deliberate, unmistakable pose, so it wins
        # over any hand shape - screaming with your hands up is still a
        # scream. Checked before the no-hands branch for that reason.
        if face_is_fresh and self.last_face[2] > SCREAM_MOUTH_OPEN:
            return "screamingCat"

        if not hand_result.hand_landmarks:
            # no hands: these are face-only poses (head turned or tilted, no
            # particular hand shape needed). Side-eye is checked first
            # because a turned head reads as the stronger intent when both
            # thresholds happen to trip.
            if face_is_fresh and abs(self.last_face[3]) > SIDE_EYE_YAW_DEG:
                return "sideEyeCat"
            if face_is_fresh and abs(self.last_face[4]) > HUH_ROLL_DEG:
                return "huhCat"
            return "default"

        hands = [classify_hand(lm) for lm in hand_result.hand_landmarks]

        if len(hands) == 2:
            if is_pointing(hands[0]) and is_pointing(hands[1]):
                avg_scale = (hands[0]["handScale"] + hands[1]["handScale"]) / 2
                tip_gap = dist(hands[0]["indexTip"], hands[1]["indexTip"]) / avg_scale
                if tip_gap < 1.4:
                    return "twoFingersTogether"

            if face_is_fresh:
                mouth_center, face_width, _, _, _, _ = self.last_face
                near_face = all(
                    dist(h["palmCenter"], mouth_center) / face_width < 2.2 for h in hands
                )
                if near_face:
                    head_top_y = mouth_center[1] - face_width * 1.1
                    both_above_head = all(h["palmCenter"][1] < head_top_y for h in hands)
                    if both_above_head:
                        return "twoHandsOnHead"
                    return "crashOutCat"

        h = hands[0]

        if h["curledCount"] == 4:
            return "fist"

        if h["thumbOut"] and h["pinkyUp"] and not h["indexUp"] and not h["middleUp"] and not h["ringUp"]:
            return "rockstar"

        # shhh / one-finger-up: a single extended index finger is a very
        # specific shape (shhh in particular = fingertip right on the
        # mouth), so it must be checked before the broader hand-covering-
        # face test below - otherwise a shhh pose (finger near the mouth)
        # gets swallowed by the "any hand near the face" check.
        if h["indexUp"] and not h["middleUp"] and not h["ringUp"] and not h["pinkyUp"]:
            if face_is_fresh:
                mouth_center, face_width, _, _, _, _ = self.last_face
                d = dist(h["indexTip"], mouth_center) / face_width
                if d < 0.55:
                    return "shhh"
            return "oneFingerUp"

        # hand covering face: the one hand we see sits roughly where the
        # face last was. Wider tolerance if the face detector has fully
        # lost the face (strong evidence of a real occlusion); tighter if
        # it's still partially tracking through the fingers.
        if face_is_fresh:
            mouth_center, face_width, _, _, _, _ = self.last_face
            d = dist(h["palmCenter"], mouth_center) / face_width
            threshold = (
                HAND_COVER_FACE_DIST_FACE_LOST
                if not self.face_seen_this_frame
                else HAND_COVER_FACE_DIST_FACE_SEEN
            )
            if d < threshold:
                return "handCoverFace"

        # open palm held out, not near the face
        if h["curledCount"] == 0:
            return "handStretchedOut"

        # hands are up but not making a specific shape - still allow a
        # strong side-eye or head-tilt read to win over an ambiguous pose.
        if face_is_fresh and abs(self.last_face[3]) > SIDE_EYE_YAW_DEG:
            return "sideEyeCat"
        if face_is_fresh and abs(self.last_face[4]) > HUH_ROLL_DEG:
            return "huhCat"

        return "default"


def load_memes():
    """Load every meme image, keyed by gesture.

    A gesture whose image files are all missing is skipped rather than
    fatal, and simply never fires (decide() falls back to default for it) -
    that way a gesture can be wired up here before its artwork exists
    without taking the whole app down. Anything missing is reported on
    stdout so it doesn't fail silently.
    """
    cache = {}
    for gesture, files in GESTURE_MEMES.items():
        if gesture in VIDEO_GESTURES:
            # videos are streamed frame-by-frame in the main loop instead
            continue
        imgs = []
        for name in files:
            img = cv2.imread(str(MEMES / name))
            if img is None:
                print(f"[meme missing] {MEMES / name} - '{gesture}' disabled until it's added")
                continue
            imgs.append(img)
        if imgs:
            cache[gesture] = imgs
    return cache


def draw_debug_hud(frame, state, gesture):
    # compact: value and its trigger threshold per line, no prose
    lines = [
        f"{gesture}",
        f"yaw  {state.last_yaw_debug:+5.1f}/{SIDE_EYE_YAW_DEG:.0f}",
        f"roll {state.last_roll_debug:+5.1f}/{HUH_ROLL_DEG:.0f}",
        f"mouth {state.last_mouth_open_debug:.2f}/{SCREAM_MOUTH_OPEN:.2f}",
        f"flow {state.last_flow_magnitude_debug:.2f}/{SPIN_MAG_THRESHOLD:.2f}",
        f"spin {state.last_flow_fraction_debug:.2f}/{SPIN_FRACTION_REQUIRED:.2f}"
        f"  pk {state.last_flow_peak_debug:.2f}",
    ]

    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1
    pad, line_h = 6, 14

    # Outline drawn as the same text, same thickness, nudged one pixel in
    # each direction. Drawing it once at a heavier thickness instead (the
    # usual trick) doesn't work here: in OpenCV thickness also widens the
    # per-glyph advance, so a thickness-3 pass ends up to ~30px wider than
    # the thickness-1 fill on these lines, and the two drift apart into
    # what looks like a second, offset copy of the readout.
    for i, line in enumerate(lines):
        y = pad + line_h * i + 10
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)):
            cv2.putText(frame, line, (pad + dx, y + dy), font, scale, (0, 0, 0), thick, cv2.LINE_AA)
        cv2.putText(frame, line, (pad, y), font, scale, (0, 255, 120), thick, cv2.LINE_AA)


def draw_landmarks(frame, hand_result):
    h, w = frame.shape[:2]
    for hand in hand_result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (80, 220, 120), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (60, 140, 255), -1)


def get_screen_size():
    """Actual (logical) display resolution, so the two windows can be sized
    to fit on screen instead of guessing.

    Asks the window server through osascript rather than tkinter: Tk 8.6
    aborts the whole process with an uncatchable ObjC NSException
    (-[NSApplication macOSVersion]) on recent macOS, and osascript runs in
    a subprocess, so any failure there can't take this process down.
    """
    try:
        out = subprocess.run(
            ["osascript", "-e", "tell application \"Finder\" to get bounds of window of desktop"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        _, _, w, h = (int(v.strip()) for v in out.split(","))
        return w, h
    except Exception:
        return 1440, 900


# Memes are shown at their own aspect ratio - never padded, never stretched,
# so there are no letterbox bars. The Meme window autosizes to whatever it's
# given, so it does change width from meme to meme, but the layout reserves
# a box to its right big enough for the widest one, so that growth always
# lands in empty reserved space rather than off screen or over the Camera
# window. Since the window is never moved after placement, dragging works.
#
# Normally a meme is drawn at the shared display height. The one exception
# is the spin video, which is far wider than any still meme - letting it
# drive the reserved width would shrink the shared height (and so the cam)
# for the whole session just for the rare moments it plays. Instead the box
# is sized for the still memes, and anything wider than the box (only the
# video) is scaled down to the box width.
def fit_meme(img, height, max_w):
    w = max(1, int(img.shape[1] * height / img.shape[0]))
    if w > max_w:
        w, height = max_w, max(1, int(img.shape[0] * max_w / img.shape[1]))
    return cv2.resize(img, (w, height))


# The web version stays smooth under load because the <video> tag renders on
# its own, independent of the JS detection loop - the loop just reads
# whatever frame happens to be current. cv2.imshow has no such decoupling:
# if capture/detect/draw/show all happen in one loop, display fps is capped
# by mediapipe's (CPU-bound, here) inference time. So capture and detection
# run in their own free-running threads, and the display loop only ever
# grabs the latest frame + latest detection result - display fps is then
# bounded by the camera, not by mediapipe.
class SharedFrame:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None

    def set(self, frame):
        with self.lock:
            self.frame = frame

    def get(self):
        with self.lock:
            return self.frame


class SharedDetection:
    def __init__(self):
        self.lock = threading.Lock()
        self.hand_result = None
        self.gesture = "default"

    def set(self, hand_result, gesture):
        with self.lock:
            self.hand_result = hand_result
            self.gesture = gesture

    def get(self):
        with self.lock:
            return self.hand_result, self.gesture


def capture_loop(cap, shared_frame, stop_event):
    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            stop_event.set()
            break
        shared_frame.set(cv2.flip(frame, 1))  # mirror, like a selfie cam


def detection_loop(
    hand_landmarker,
    face_landmarker,
    shared_frame,
    shared_detection,
    state,
    memes,
    flow_log,
    stop_event,
):
    prev_flow_gray = None
    current_gesture = "default"
    candidate_gesture = "default"
    candidate_streak = 0
    last_non_default_at = time.time() * 1000
    start_time = time.time()

    while not stop_event.is_set():
        frame = shared_frame.get()
        if frame is None:
            time.sleep(0.005)
            continue

        magnitude, coherence, prev_flow_gray = frame_flow_signal(frame, prev_flow_gray)
        state.update_flow(magnitude, coherence)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.time() - start_time) * 1000)

        hand_result = hand_landmarker.detect_for_video(mp_image, ts_ms)
        face_result = face_landmarker.detect_for_video(mp_image, ts_ms)
        state.update_face(face_result)

        gesture = state.decide(hand_result)

        # logged after detection, so every column is from this same frame -
        # the face columns are here to tune the mouth/roll thresholds
        # against real recordings rather than by guesswork
        flow_log.write(
            f"{time.time() * 1000:.0f},{magnitude:.4f},{coherence:.4f},"
            f"{state.last_flow_score_debug:.4f},{state.last_flow_fraction_debug:.4f},"
            f"{state.last_flow_peak_debug:.4f},"
            f"{state.last_mouth_open_debug:.4f},{state.last_yaw_debug:.2f},"
            f"{state.last_roll_debug:.2f},{int(state.face_seen_this_frame)},"
            f"{gesture},{current_gesture}\n"
        )

        now = time.time() * 1000
        if gesture == candidate_gesture:
            candidate_streak += 1
        else:
            candidate_gesture = gesture
            candidate_streak = 1

        if candidate_streak >= STABLE_FRAMES_REQUIRED and gesture != current_gesture:
            current_gesture = gesture
            if gesture in VIDEO_GESTURES:
                if gesture == "spinCat":
                    memes["_spin_restart"] = True
            elif gesture in memes:
                memes["_current"] = random.choice(memes[gesture])
            # else: gesture is wired up but its artwork isn't in memes/ yet
            # (load_memes said so at startup). Still switch to it so the
            # readout names it - that's what makes the detection tunable
            # before the art exists - and leave the meme window showing
            # whatever it had.

        if gesture != "default":
            last_non_default_at = now
        elif now - last_non_default_at > DEFAULT_FALLBACK_MS and current_gesture != "default":
            current_gesture = "default"
            memes["_current"] = random.choice(memes["default"])

        shared_detection.set(hand_result, current_gesture)


def main():
    hand_landmarker = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "hand_landmarker.task")),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
        )
    )
    face_landmarker = FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "face_landmarker.task")),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
    )

    memes = load_memes()
    # widest aspect ratio among the still memes - the reserved box is sized
    # to this, so every still meme fits it at full height. The spin video is
    # deliberately excluded: it's much wider than any of these, and letting
    # it set the box would cost height (and cam size) all session long for
    # the rare moments it plays; it gets scaled down to the box instead.
    # Computed before the bookkeeping keys below are added, while every
    # value is still a list of images.
    widest_meme_aspect = max(img.shape[1] / img.shape[0] for imgs in memes.values() for img in imgs)
    memes["_current"] = random.choice(memes["default"])
    memes["_spin_restart"] = False

    # every frame's flow numbers get logged here, timestamped - so we can
    # look at exactly what a real, full-effort spin looked like afterward
    # instead of trying to read a jittery number while dizzy.
    flow_log_path = ROOT / "flow_debug_log.csv"
    flow_log = open(flow_log_path, "w", buffering=1)  # line-buffered so data survives a hard kill
    flow_log.write(
        "t_ms,magnitude,coherence,score,fraction,peak_2s,"
        "mouth_open,yaw,roll,face_seen,gesture_raw,gesture_shown\n"
    )

    spin_video_cap = cv2.VideoCapture(str(MEMES / GESTURE_MEMES["spinCat"][0]))
    if not spin_video_cap.isOpened():
        raise FileNotFoundError(f"missing meme file: {MEMES / GESTURE_MEMES['spinCat'][0]}")

    def next_spin_frame():
        ok, vframe = spin_video_cap.read()
        if not ok:
            spin_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, vframe = spin_video_cap.read()
        return vframe

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0)")
    # ask for a proper resolution and fps - an unconfigured VideoCapture
    # often defaults to a low-res mode. MJPG gives the camera more bandwidth
    # headroom to hit both the resolution and the fps target.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always grab the newest frame, not a queued stale one

    cv2.namedWindow("Camera")
    cv2.namedWindow("Meme")
    cv2.moveWindow("Camera", 40, 80)

    state = GestureState()
    shared_frame = SharedFrame()
    shared_detection = SharedDetection()
    stop_event = threading.Event()

    cap_thread = threading.Thread(target=capture_loop, args=(cap, shared_frame, stop_event), daemon=True)
    det_thread = threading.Thread(
        target=detection_loop,
        args=(
            hand_landmarker,
            face_landmarker,
            shared_frame,
            shared_detection,
            state,
            memes,
            flow_log,
            stop_event,
        ),
        daemon=True,
    )
    cap_thread.start()
    det_thread.start()

    try:
        # wait for the first camera frame so the window doesn't flash empty
        while shared_frame.get() is None and not stop_event.is_set():
            time.sleep(0.01)

        # Capture runs at 1280x720 for quality and detection accuracy, but
        # that's far too wide to show next to a meme on this screen, so
        # both windows share one display height, picked once so that the
        # cam and the widest meme fit side by side. The cam window is a
        # fixed size for the session; the meme window varies in width with
        # each meme (shown at its own aspect, so no bars), but the space
        # reserved to its right fits the widest one, so it can never grow
        # off screen or over the cam.
        first_frame = shared_frame.get()
        cam_aspect = first_frame.shape[1] / first_frame.shape[0]
        screen_w, screen_h = get_screen_size()
        LEFT, GAP, RIGHT, TOP, BOTTOM = 12, 16, 12, 60, 40
        # SIZE_SCALE pushes past the height at which cam and widest meme both
        # fit at full size. The leftover width below absorbs it: the cam gets
        # the full increase, and the widest memes give back a few pixels of
        # width (scaled by fit_meme) rather than the whole layout staying
        # small. Trimmed margins above supply most of the extra room.
        SIZE_SCALE = 1.04
        display_h = min(
            screen_h - TOP - BOTTOM,
            int(SIZE_SCALE * (screen_w - LEFT - GAP - RIGHT) / (cam_aspect + widest_meme_aspect)),
        )
        cam_w, cam_h = int(cam_aspect * display_h), display_h
        # whatever width is left after the cam - guarantees the meme window,
        # however wide the meme, always lands on screen
        meme_max_w = screen_w - LEFT - cam_w - GAP - RIGHT

        cv2.moveWindow("Camera", LEFT, TOP)
        cv2.moveWindow("Meme", LEFT + cam_w + GAP, TOP)

        while not stop_event.is_set():
            frame = shared_frame.get()
            if frame is None:
                continue
            # scale to display size FIRST, then draw - the overlays used to
            # be drawn on the full 1280-wide frame and shrunk with it, which
            # made the readout text noticeably smaller than intended (and
            # meant drawing more pixels than needed). Landmarks use
            # normalized coords, so they scale to any size.
            frame = cv2.resize(frame, (cam_w, cam_h))

            hand_result, current_gesture = shared_detection.get()
            if hand_result is not None:
                draw_landmarks(frame, hand_result)
            draw_debug_hud(frame, state, current_gesture)

            if current_gesture == "spinCat":
                if memes["_spin_restart"]:
                    spin_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    memes["_spin_restart"] = False
                vframe = next_spin_frame()
                meme_img = vframe if vframe is not None else memes["_current"]
            else:
                meme_img = memes["_current"]

            # own aspect ratio, no bars; only the spin video hits max_w
            meme_view = fit_meme(meme_img, display_h, meme_max_w)

            cv2.imshow("Camera", frame)
            cv2.imshow("Meme", meme_view)

            # no delay beyond what's needed to pump the GUI event loop - the
            # display rate is bounded by the camera thread, not by this wait
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        stop_event.set()
        cap_thread.join(timeout=1)
        det_thread.join(timeout=1)
        cap.release()
        spin_video_cap.release()
        flow_log.close()
        cv2.destroyAllWindows()
        hand_landmarker.close()
        face_landmarker.close()


if __name__ == "__main__":
    main()
