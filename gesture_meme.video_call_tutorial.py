"""
Webcam gesture -> meme detector (desktop version).

Opens a single resizable window, video-call style: the cat meme matching
your current gesture fills the whole window, with your own webcam feed
(hand landmarks drawn on top) as a small picture-in-picture bubble in the
bottom-right corner. Maximize/resize the window and the meme scales to fill
it.

Click the "Tutorial" button in the top-right corner (or press t) to switch
into a guided slideshow of every gesture and how to trigger it, with
Back/Play-Pause/Next buttons (or p/space/n) to step through at your own
pace instead of hunting for how to get each meme to show up.

Gestures:
  rockstar / shaka  -> memes/cat.jpg
  default (no hand) -> memes/pokercat.jpg
  one finger up     -> memes/profcat.jpg, memes/professorcat.jpg
  fist / punch      -> memes/punchcat.jpg
  shhh              -> memes/shhcat.jpg
  two fingers together (both hands, tips touching) -> memes/uwucat.jpg, memes/uwucatt.jpg,
                                                        memes/fingers together muehehe .jpg
  hand covering face -> memes/hand cover face .jpg
  crash-out cat (two CLENCHED FISTS up beside the face)   -> memes/crashout cat .jpg
  two hands on head                                        -> memes/two hands on head .jpg
  dance cat (two open palms, one near top of screen, one near bottom)  -> memes/two palms up.mov (video)
  hand stretched out, palm facing camera (open hand)       -> memes/hand stretched out, palm facing up .jpg
  side eye (head turned to the side)                       -> memes/side eye cat.jpg
  slight side eye (head tilted DOWN, not turned)            -> memes/side eye.png
  mouth wide open, WITH a hand visible somewhere in frame  -> memes/laugh and point .jpg
  huh cat (mouth open AND eyes wide, no hands)               -> memes/huh.png
  shrug cat (both hands up and out to the sides)             -> memes/iunno cat.jpg
  spin cat (spinning fast in your chair)                   -> memes/spin cat.mov (plays as a video)

Facial expressions (mouth-open above) use MediaPipe's face blendshapes -
pre-computed expression scores (0-1) that come from the same face model,
rather than hand-rolled landmark geometry. Several more scores are already
being read and shown on the debug HUD (smile, brow-raise, wink) but not
wired to a meme yet - see the "facial expression thresholds" section below
to add more once you've picked images for them.

The Camera window shows a live debug readout (head yaw, and optical-flow
magnitude/coherence, vs. their trigger thresholds) in the top-left corner so
side-eye and spin can both be tuned by eye - see SIDE_EYE_YAW_DEG and
SPIN_FLOW_SCORE_THRESHOLD below.

Press q or ESC to quit. Press d to toggle the debug HUD (off by default).
"""



import os

# quiet mediapipe/absl/TFLite's INFO/WARNING spam (XNNPACK delegate notices,
# "feedback manager" warnings, etc.) - must be set before mediapipe is imported
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import math
import random
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
    "sideEyeDownCat": ["side eye.png"],
    "mouthOpenCat": ["laugh and point .jpg"],
    "huhCat": ["huh.png"],
    "danceCat": ["two palms up.mov"],
    "spinCat": ["spin cat.mov"],
}

# gestures whose meme is a video, not a still image
VIDEO_GESTURES = {"spinCat", "danceCat"}

# display name + "how to trigger" caption for tutorial mode, in the order
# they're stepped through. (gesture, title, how-to-trigger)
TUTORIAL_STEPS = [
    ("default", "Default", "Nothing in particular, hands down"),
    ("rockstar", "Rockstar", "Thumb + pinky out"),
    ("oneFingerUp", "One Finger Up", "Index finger only, held away from your face"),
    ("fist", "Fist", "One hand, all four fingers curled"),
    ("shhh", "Shhh", "Index finger only, tip resting on your mouth"),
    ("twoFingersTogether", "Fingers Together (muehehe)", "Both hands up, index fingers only, tips touching"),
    ("handCoverFace", "Kidnap Cat", "Any hand shape sitting where your face just was"),
    ("crashOutCat", "Crash Out Cat", "Both hands up beside your face, not above your head"),
    ("twoHandsOnHead", "Devastated Cat", "Both hands up, above the top of your head"),
    ("handStretchedOut", "i HAVE NO MONIES", "One open palm, all fingers extended, away from your face"),
    ("sideEyeCat", "Side Eye", "Just turn your head"),
    ("sideEyeDownCat", "Judgy Cat (munching)", "Look down slightly sideways"),
    ("mouthOpenCat", "Laugh & Point", "Mouth wide open, with a hand visible somewhere in frame"),
    ("huhCat", "Huh", "Open your mouth and be shocked"),
    ("danceCat", "Dance Cat", "Both hands open, one near the top of the screen, one near the bottom"),
    ("spinCat", "O I I.A I (Spin)", "Spin spin spin!"),
]

# gestures made with a single hand - the meme image mirrors to whichever
# side of the frame that hand is on, so "your left hand up" shows a
# left-facing cat instead of always the same default-facing image.
MIRROR_BY_HAND_GESTURES = {
    "fist", "rockstar", "shhh", "oneFingerUp", "handStretchedOut",
    "handCoverFace", "mouthOpenCat",
}

# gestures driven by head turn (yaw) instead of a hand - mirror the same way,
# based on which way the head is turned rather than hand position.
MIRROR_BY_YAW_GESTURES = {"sideEyeCat", "sideEyeDownCat"}

STABLE_FRAMES_REQUIRED = 5
DEFAULT_FALLBACK_MS = 600
FACE_STALE_MS = 1200

# how far the head has to turn (yaw, in degrees, from MediaPipe's own head
# pose estimate - not a hand-rolled distance heuristic) to count as a
# side-eye look. Watch the live "yaw" readout in the Camera window while
# turning your head to find the right value for you.
SIDE_EYE_YAW_DEG = 15.0

# same idea, but for tilting the head DOWN (pitch) instead of turning it
# sideways (yaw) - "slight side eye" when you look down. Uses the same
# transformation-matrix approach as yaw (pitch_from_transform_matrix below),
# NOT tested against real degrees yet the way SIDE_EYE_YAW_DEG was (that one
# was validated live against the debug HUD's yaw readout) - watch the new
# "pitch" line on the debug HUD while looking down and adjust this to match.
SIDE_EYE_DOWN_PITCH_DEG = 12.0

# huh cat: mouth open AND eyes wide (distinct from mouthOpenCat, which is
# jawOpen alone). Uses the eyeWideLeft/eyeWideRight and jawOpen blendshapes.
# Real test readings: (jaw=0.05, eye=0.45), (jaw=0.4-0.5, eye=0.05) - a
# natural "huh" face doesn't reliably max out both blendshapes at once. At
# 0.03, that last eye=0.05 reading only clears the bar by 0.02 - since the
# app requires 5 CONSECUTIVE frames above threshold before switching, and
# blendshape scores jitter frame to frame, that thin a margin likely kept
# dipping back under and resetting the streak. Lowered further for a real
# buffer against that jitter.
# huhCat gets its own jaw threshold instead of sharing
# MOUTH_OPEN_JAW_THRESHOLD with mouthOpenCat, so tuning this doesn't also
# make mouthOpenCat (which wasn't broken) trigger too easily.
EYE_WIDE_THRESHOLD = 0.01
HUH_JAW_THRESHOLD = 0.03

# danceCat: one open hand near the top of the screen, the other near the
# bottom - absolute frame position (0.0 = top edge, 1.0 = bottom edge), not
# relative to your face. Untested against real numbers - watch each hand's
# y position (not currently on the debug HUD; add one if this needs tuning)
# and adjust these zones if they feel too tight/loose.
DANCE_TOP_ZONE_Y = 0.35
DANCE_BOTTOM_ZONE_Y = 0.65

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
SPIN_MAG_THRESHOLD = 0.65  # per-frame magnitude counted as "elevated" for the fraction test
# ^ was 0.8 - real recorded data from a genuine spin attempt (flow_debug_log.csv)
# showed a classic wind-up/sustain/wind-down curve with magnitude sitting in the
# 1.0-1.7 band for over a second, but its fraction still capped at 0.52 (just
# under the 0.55 SPIN_FRACTION_REQUIRED below) because 0.8 was too strict a bar
# for that sustained band to clear often enough. False-positive bursts (quick
# leans/turns) spike much higher (3+), so lowering this doesn't stop those from
# still triggering too - that's a separate problem, not fixed by this change.
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

# facial expression thresholds - these read MediaPipe's face blendshapes
# (output_face_blendshapes=True below), which are pre-computed 0-1 scores
# per expression from the face model itself, not hand-rolled geometry like
# the hand gestures above. Only "mouth open" has a meme wired up so far
# (memes/laugh and point .jpg fit it well and was already sitting unused).
# smileScore/browRaiseScore/winkScore are already being read and shown on
# the debug HUD - once you've got images that fit them, add a check for
# each in GestureState.decide() the same way MOUTH_OPEN_JAW_THRESHOLD is
# used below, and add the gesture name -> meme file(s) to GESTURE_MEMES.
MOUTH_OPEN_JAW_THRESHOLD = 0.5   # jawOpen score; watch the debug HUD while opening your mouth to tune
SMILE_THRESHOLD = 0.6            # max(mouthSmileLeft, mouthSmileRight)
BROW_RAISE_THRESHOLD = 0.5       # browInnerUp
WINK_THRESHOLD = 0.5             # one eye's blink score high, the other's low - see wink_score()

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


def blendshape_scores(face_result):
    """Pull MediaPipe's face blendshapes into a plain {name: score} dict.
    Returns {} if blendshapes weren't returned this frame (no face, or the
    option wasn't enabled)."""
    if not face_result.face_blendshapes:
        return {}
    return {b.category_name: b.score for b in face_result.face_blendshapes[0]}


def wink_score(scores):
    """One eye closed, the other open - the absolute gap between the two
    blink scores, only counted once at least one eye is clearly closing
    (otherwise two half-lowered eyes would also read as a 'gap')."""
    left, right = scores.get("eyeBlinkLeft", 0.0), scores.get("eyeBlinkRight", 0.0)
    if max(left, right) < WINK_THRESHOLD:
        return 0.0
    return abs(left - right)


def eye_wide_score(scores):
    """max(eyeWideLeft, eyeWideRight) - how wide the eyes are open, for
    huhCat (mouth open AND eyes wide, distinct from  mouthOpenCat which is
    mouth-open alone)."""
    return max(scores.get("eyeWideLeft", 0.0), scores.get("eyeWideRight", 0.0))


def yaw_from_transform_matrix(matrix):
    """Extract the head's left/right turn angle (yaw, degrees) from
    MediaPipe's facial transformation matrix - its own estimate of head
    pose, far more robust than trying to infer turn from landmark
    distances."""
    r = np.asarray(matrix)[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy < 1e-6:
        return 0.0
    yaw = math.atan2(-r[2, 0], sy)
    return math.degrees(yaw)


def pitch_from_transform_matrix(matrix):
    """Extract the head's up/down tilt angle (pitch, degrees) from the same
    transformation matrix as yaw above, using the analogous formula for the
    other axis of the same rotation matrix. Unlike SIDE_EYE_YAW_DEG, this
    hasn't been validated against real degrees by watching someone actually
    turn - watch the new 'pitch' line on the debug HUD while looking down
    to confirm the sign is right (should go positive) and find your real
    threshold, the same way yaw was originally tuned."""
    r = np.asarray(matrix)[:3, :3]
    pitch = math.atan2(r[2, 1], r[2, 2])
    return math.degrees(pitch)


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
        self.last_face = None  # (mouth_center, face_width, mouth_open, yaw_deg, t)
        self.face_seen_this_frame = False
        self.last_yaw_debug = 0.0
        self.flow_history = []  # [(t, magnitude), ...] trailing samples, for the fraction-above trigger
        self.flow_peak_history = []  # [(t, score), ...] longer trailing window, for the readable peak display
        self.last_flow_magnitude_debug = 0.0
        self.last_flow_coherence_debug = 0.0
        self.last_flow_score_debug = 0.0
        self.last_flow_peak_debug = 0.0
        self.last_flow_fraction_debug = 0.0
        self.last_blendshapes = {}  # {category_name: score}, refreshed whenever a face is seen
        self.last_jaw_open_debug = 0.0
        self.last_smile_debug = 0.0
        self.last_brow_raise_debug = 0.0
        self.last_wink_debug = 0.0
        self.last_eye_wide_debug = 0.0
        self.last_pitch_debug = 0.0

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
            mouth_center = (upper_lip + lower_lip) / 2
            face_width = dist(right_cheek, left_cheek)
            mouth_open = dist(upper_lip, lower_lip) / face_width

            yaw_deg = 0.0
            pitch_deg = 0.0
            if face_result.facial_transformation_matrixes:
                matrix = face_result.facial_transformation_matrixes[0]
                yaw_deg = yaw_from_transform_matrix(matrix)
                pitch_deg = pitch_from_transform_matrix(matrix)

            self.last_face = (mouth_center, face_width, mouth_open, yaw_deg, now)
            self.last_yaw_debug = yaw_deg
            self.last_pitch_debug = pitch_deg

            self.last_blendshapes = blendshape_scores(face_result)
            self.last_jaw_open_debug = self.last_blendshapes.get("jawOpen", 0.0)
            self.last_smile_debug = max(
                self.last_blendshapes.get("mouthSmileLeft", 0.0),
                self.last_blendshapes.get("mouthSmileRight", 0.0),
            )
            self.last_brow_raise_debug = self.last_blendshapes.get("browInnerUp", 0.0)
            self.last_wink_debug = wink_score(self.last_blendshapes)
            self.last_eye_wide_debug = eye_wide_score(self.last_blendshapes)
        self.face_seen_this_frame = saw_face

    def decide(self, hand_result):
        now = time.time() * 1000
        face_is_fresh = self.last_face is not None and now - self.last_face[4] < FACE_STALE_MS

        # spinning in the chair beats everything else, hands included.
        if self.is_spinning(now):
            return "spinCat"

        if not hand_result.hand_landmarks:
            # no hands: side-eye and huh are both face-only poses, and
            # BOTH require no hands visible - mouthOpenCat moved out of
            # this branch entirely (see below), specifically so it can
            # never clash with huhCat: huhCat needs mouth-open with no
            # hand, mouthOpenCat now needs mouth-open WITH a hand. They're
            # mutually exclusive by construction, not by priority-ordering.
            if (
                face_is_fresh
                and self.last_jaw_open_debug > HUH_JAW_THRESHOLD
                and self.last_eye_wide_debug > EYE_WIDE_THRESHOLD
            ):
                return "huhCat"
            if face_is_fresh and abs(self.last_face[3]) > SIDE_EYE_YAW_DEG:
                return "sideEyeCat"
            if face_is_fresh and self.last_pitch_debug > SIDE_EYE_DOWN_PITCH_DEG:
                return "sideEyeDownCat"
            return "default"

        # mouthOpenCat: mouth open AND a hand visible somewhere in frame -
        # any hand shape counts, this isn't about what the hand is doing,
        # just that one's present. Checked before the hand-shape-specific
        # branches below so an open mouth with a hand up (eating, talking
        # with your hands, etc.) reads as this rather than whatever shape
        # the hand happens to be making.
        if face_is_fresh and self.last_jaw_open_debug > MOUTH_OPEN_JAW_THRESHOLD:
            return "mouthOpenCat"

        hands = [classify_hand(lm) for lm in hand_result.hand_landmarks]

        if len(hands) == 2:
            if is_pointing(hands[0]) and is_pointing(hands[1]):
                avg_scale = (hands[0]["handScale"] + hands[1]["handScale"]) / 2
                tip_gap = dist(hands[0]["indexTip"], hands[1]["indexTip"]) / avg_scale
                if tip_gap < 1.4:
                    return "twoFingersTogether"

            if face_is_fresh:
                mouth_center, face_width, _, _, _ = self.last_face
                near_face = all(
                    dist(h["palmCenter"], mouth_center) / face_width < 2.2 for h in hands
                )
                if near_face:
                    head_top_y = mouth_center[1] - face_width * 1.1
                    both_above_head = all(h["palmCenter"][1] < head_top_y for h in hands)
                    if both_above_head:
                        return "twoHandsOnHead"
                    # crashOutCat requires both hands to actually be
                    # clenched fists - an open hand near the face falls
                    # through instead of getting swallowed by this.
                    both_fists = all(h["curledCount"] == 4 for h in hands)
                    if both_fists:
                        return "crashOutCat"

            # danceCat: both hands showing open palms (fingers spread - the
            # practical proxy for "palm facing camera", since MediaPipe
            # doesn't give hand orientation directly), with one hand near
            # the TOP of the screen and the other near the BOTTOM -
            # absolute frame position, not relative to your face, and it
            # doesn't matter which hand is on top. Untested against real
            # numbers - watch each hand's y position and adjust
            # DANCE_TOP_ZONE_Y/DANCE_BOTTOM_ZONE_Y if the zones feel wrong.
            both_open = all(h["curledCount"] == 0 for h in hands)
            if both_open:
                ys = sorted(h["palmCenter"][1] for h in hands)
                one_near_top = ys[0] < DANCE_TOP_ZONE_Y
                one_near_bottom = ys[1] > DANCE_BOTTOM_ZONE_Y
                if one_near_top and one_near_bottom:
                    return "danceCat"

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
                mouth_center, face_width, _, _, _ = self.last_face
                d = dist(h["indexTip"], mouth_center) / face_width
                if d < 0.55:
                    return "shhh"
            return "oneFingerUp"

        # hand covering face: the one hand we see sits roughly where the
        # face last was. Wider tolerance if the face detector has fully
        # lost the face (strong evidence of a real occlusion); tighter if
        # it's still partially tracking through the fingers.
        if face_is_fresh:
            mouth_center, face_width, _, _, _ = self.last_face
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
        # strong side-eye read to win over an ambiguous hand pose.
        if face_is_fresh and abs(self.last_face[3]) > SIDE_EYE_YAW_DEG:
            return "sideEyeCat"

        return "default"


def load_memes():
    cache = {}
    for gesture, files in GESTURE_MEMES.items():
        if gesture in VIDEO_GESTURES:
            # videos are streamed frame-by-frame in the main loop instead
            continue
        imgs = []
        for name in files:
            img = cv2.imread(str(MEMES / name))
            if img is None:
                raise FileNotFoundError(f"missing meme file: {MEMES / name}")
            imgs.append(img)
        cache[gesture] = imgs
    return cache


def draw_debug_hud(frame, state, gesture):
    lines = [
        f"gesture: {gesture}",
        f"yaw: {state.last_yaw_debug:+.1f} deg  (side-eye thr +/-{SIDE_EYE_YAW_DEG:.1f})",
        f"flow mag: {state.last_flow_magnitude_debug:.2f}  (thr {SPIN_MAG_THRESHOLD:.2f})",
        f"spin fraction (2.2s window): {state.last_flow_fraction_debug:.2f}  (thr {SPIN_FRACTION_REQUIRED:.2f})",
        f"peak score (last 2s): {state.last_flow_peak_debug:.2f}  <- read this AFTER you stop spinning",
        f"jawOpen: {state.last_jaw_open_debug:.2f}  eyeWide: {state.last_eye_wide_debug:.2f}  "
        f"(huh needs both > {HUH_JAW_THRESHOLD:.2f}/{EYE_WIDE_THRESHOLD:.2f})",
        f"smile: {state.last_smile_debug:.2f}  browRaise: {state.last_brow_raise_debug:.2f}  "
        f"wink: {state.last_wink_debug:.2f}  <- not wired to a meme yet",
        f"pitch: {state.last_pitch_debug:+.1f} deg  (side-eye-down thr {SIDE_EYE_DOWN_PITCH_DEG:.1f}, unvalidated - watch this while looking down)",
    ]
    for i, line in enumerate(lines):
        y = 18 + i * 16
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 120), 1, cv2.LINE_AA)


def draw_landmarks(frame, hand_result):
    h, w = frame.shape[:2]
    for hand in hand_result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (80, 220, 120), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (60, 140, 255), -1)


def fit_to_height(img, height):
    h, w = img.shape[:2]
    scale = height / h
    return cv2.resize(img, (int(w * scale), height))


# video-call style layout: the meme shows at its real aspect ratio, fit
# inside the window with letterbox bars on whichever sides don't match
# (like CSS object-fit: contain) rather than being cropped, with the webcam
# feed as a small picture-in-picture bubble over it - like the "you" tile
# on a call.
PIP_SIZE_FRACTION = 0.14
PIP_MARGIN = 18
PIP_BORDER_THICKNESS = 3
PIP_BORDER_COLOR = (240, 240, 240)
LETTERBOX_COLOR = (18, 18, 18)


def contain_fit(img, target_w, target_h):
    """Resize img to fit entirely inside (target_w, target_h) preserving its
    aspect ratio, centered on a solid background - like CSS
    object-fit: contain. Whatever doesn't fill target_w/target_h stays as
    letterbox/pillarbox bars instead of cropping the image."""
    h, w = img.shape[:2]
    if target_w <= 0 or target_h <= 0:
        return img
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, math.floor(w * scale)), max(1, math.floor(h * scale))
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.full((target_h, target_w, 3), LETTERBOX_COLOR, dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def compose_video_call_view(meme_source, cam_frame, canvas_w, canvas_h):
    """Build the single composited frame shown in the window: the meme at
    its real aspect ratio (letterboxed to fit), with the webcam feed as a
    small bubble in the bottom-right corner, video-call style."""
    canvas = contain_fit(meme_source, canvas_w, canvas_h)

    pip_w = max(1, int(canvas_w * PIP_SIZE_FRACTION))
    fh, fw = cam_frame.shape[:2]
    pip_h = max(1, int(pip_w * fh / fw))
    if pip_w >= canvas_w or pip_h >= canvas_h:
        return canvas

    thumb = cv2.resize(cam_frame, (pip_w, pip_h))
    x0 = canvas_w - pip_w - PIP_MARGIN
    y0 = canvas_h - pip_h - PIP_MARGIN
    canvas[y0:y0 + pip_h, x0:x0 + pip_w] = thumb
    cv2.rectangle(
        canvas,
        (x0 - PIP_BORDER_THICKNESS, y0 - PIP_BORDER_THICKNESS),
        (x0 + pip_w + PIP_BORDER_THICKNESS - 1, y0 + pip_h + PIP_BORDER_THICKNESS - 1),
        PIP_BORDER_COLOR,
        PIP_BORDER_THICKNESS,
    )
    return canvas


# ---- tutorial mode UI: on-screen buttons + caption bar -------------------
BUTTON_COLOR = (60, 60, 60)
BUTTON_HOVER_COLOR = (90, 90, 90)
BUTTON_TEXT_COLOR = (240, 240, 240)
BUTTON_BAR_COLOR = (25, 25, 25)
CAPTION_BAR_COLOR = (25, 25, 25)
CAPTION_TITLE_COLOR = (255, 255, 255)
CAPTION_SUBTITLE_COLOR = (170, 220, 255)


def draw_button(canvas, rect, label, pressed=False):
    x0, y0, x1, y1 = rect
    color = BUTTON_HOVER_COLOR if pressed else BUTTON_COLOR
    cv2.rectangle(canvas, (x0, y0), (x1, y1), color, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), BUTTON_TEXT_COLOR, 1, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    tx = x0 + (x1 - x0 - tw) // 2
    ty = y0 + (y1 - y0 + th) // 2
    cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, BUTTON_TEXT_COLOR, 1, cv2.LINE_AA)


def draw_tutorial_ui(canvas, ui_state, tutorial_mode, tutorial_playing, step):
    """Draws the tutorial toggle button (always visible), and when tutorial
    mode is on, the caption bar (gesture name + how to trigger it) and the
    prev/play-pause/next/exit controls. Stores each button's screen rect in
    ui_state['buttons'] so the mouse callback can hit-test clicks against
    this frame's actual layout."""
    canvas_h, canvas_w = canvas.shape[:2]
    buttons = {}

    toggle_w, toggle_h = 150, 40
    toggle_rect = (canvas_w - toggle_w - 14, 14, canvas_w - 14, 14 + toggle_h)
    draw_button(canvas, toggle_rect, "Tutorial: ON" if tutorial_mode else "Tutorial: OFF", pressed=tutorial_mode)
    buttons["toggle"] = toggle_rect

    if tutorial_mode:
        bar_h = 64
        bar_y0 = canvas_h - bar_h
        cv2.rectangle(canvas, (0, bar_y0), (canvas_w, canvas_h), BUTTON_BAR_COLOR, -1)

        btn_w, btn_h, gap = 90, 40, 14
        labels = [("prev", "<< Back"), ("play_pause", "||" if tutorial_playing else ">"), ("next", "Next >>")]
        total_w = len(labels) * btn_w + (len(labels) - 1) * gap
        x = (canvas_w - total_w) // 2
        y0 = bar_y0 + (bar_h - btn_h) // 2
        for name, label in labels:
            rect = (x, y0, x + btn_w, y0 + btn_h)
            draw_button(canvas, rect, label)
            buttons[name] = rect
            x += btn_w + gap

        gesture_key, title, howto = step
        caption_h = 56
        cap_y0 = bar_y0 - caption_h
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, cap_y0), (canvas_w, bar_y0), CAPTION_BAR_COLOR, -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
        cv2.putText(canvas, title, (20, cap_y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CAPTION_TITLE_COLOR, 2, cv2.LINE_AA)
        cv2.putText(canvas, howto, (20, cap_y0 + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CAPTION_SUBTITLE_COLOR, 1, cv2.LINE_AA)

    ui_state["buttons"] = buttons


def tutorial_mouse_callback(event, x, y, flags, ui_state):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    for name, (x0, y0, x1, y1) in ui_state.get("buttons", {}).items():
        if x0 <= x <= x1 and y0 <= y <= y1:
            ui_state["clicked"] = name
            return


def should_mirror(gesture, hand_result, state):
    """Whether the meme image should be horizontally flipped to match which
    side of the frame the gesture is happening on. The camera frame is
    already mirrored (selfie view) before MediaPipe sees it, so landmark
    x/yaw here line up with what you see on screen: x < 0.5 or a negative
    yaw is the left side of the frame, same side your own left hand/turn
    shows up on. If a gesture flips the wrong way for you, just swap the
    comparison for that branch."""
    if gesture in MIRROR_BY_YAW_GESTURES:
        return state.last_yaw_debug < 0

    if gesture in MIRROR_BY_HAND_GESTURES and hand_result.hand_landmarks:
        landmarks = hand_result.hand_landmarks[0]
        avg_x = sum(lm.x for lm in landmarks) / len(landmarks)
        return avg_x < 0.5

    return False


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
            output_face_blendshapes=True,
        )
    )

    memes = load_memes()

    # every frame's flow numbers get logged here, timestamped - so we can
    # look at exactly what a real, full-effort spin looked like afterward
    # instead of trying to read a jittery number while dizzy.
    flow_log_path = ROOT / "flow_debug_log.csv"
    flow_log = open(flow_log_path, "w", buffering=1)  # line-buffered so data survives a hard kill
    flow_log.write("t_ms,magnitude,coherence,score,fraction,peak_2s,gesture\n")

    # one VideoCapture per video gesture, keyed by gesture name - generic so
    # any gesture added to VIDEO_GESTURES gets playback/looping for free
    # without new code here.
    video_caps = {}
    for video_gesture in VIDEO_GESTURES:
        video_path = MEMES / GESTURE_MEMES[video_gesture][0]
        cap_for_gesture = cv2.VideoCapture(str(video_path))
        if not cap_for_gesture.isOpened():
            raise FileNotFoundError(f"missing meme file: {video_path}")
        video_caps[video_gesture] = cap_for_gesture

    def next_video_frame(gesture_name):
        vcap = video_caps[gesture_name]
        ok, vframe = vcap.read()
        if not ok:
            vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, vframe = vcap.read()
        return vframe

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0)")

    WINDOW_NAME = "Meow Meow Cam"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    ui_state = {"buttons": {}, "clicked": None}
    cv2.setMouseCallback(WINDOW_NAME, tutorial_mouse_callback, ui_state)

    state = GestureState()
    current_gesture = "default"
    candidate_gesture = "default"
    candidate_streak = 0
    last_non_default_at = time.time() * 1000
    current_meme = random.choice(memes["default"])
    prev_flow_gray = None
    show_hud = False  # press 'd' to toggle the debug readout back on
    canvas_size = (1280, 720)  # fallback used if the window is momentarily minimized/unreadable

    tutorial_mode = False
    tutorial_index = 0
    tutorial_playing = True
    tutorial_step_ms = 3000
    last_tutorial_advance = time.time() * 1000
    tutorial_cached_image = None  # cached still image (ndarray) for the current tutorial step; None forces a re-pick
    tutorial_video_started = False  # whether the current step's video clip has been rewound to frame 0 yet

    start_time = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror, like a selfie cam

            magnitude, coherence, prev_flow_gray = frame_flow_signal(frame, prev_flow_gray)
            state.update_flow(magnitude, coherence)
            flow_log.write(
                f"{time.time() * 1000:.0f},{magnitude:.4f},{coherence:.4f},"
                f"{state.last_flow_score_debug:.4f},{state.last_flow_fraction_debug:.4f},"
                f"{state.last_flow_peak_debug:.4f},{current_gesture}\n"
            )

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.time() - start_time) * 1000)

            hand_result = hand_landmarker.detect_for_video(mp_image, ts_ms)
            face_result = face_landmarker.detect_for_video(mp_image, ts_ms)
            state.update_face(face_result)

            gesture = state.decide(hand_result)

            now = time.time() * 1000
            if gesture == candidate_gesture:
                candidate_streak += 1
            else:
                candidate_gesture = gesture
                candidate_streak = 1

            if candidate_streak >= STABLE_FRAMES_REQUIRED and gesture != current_gesture:
                current_gesture = gesture
                if gesture not in VIDEO_GESTURES:
                    current_meme = random.choice(memes[gesture])
                else:
                    video_caps[gesture].set(cv2.CAP_PROP_POS_FRAMES, 0)

            if gesture != "default":
                last_non_default_at = now
            elif now - last_non_default_at > DEFAULT_FALLBACK_MS and current_gesture != "default":
                current_gesture = "default"
                current_meme = random.choice(memes["default"])

            draw_landmarks(frame, hand_result)

            # --- tutorial mode: handle button clicks, then autoplay ---
            clicked = ui_state.pop("clicked", None)
            if clicked == "toggle":
                tutorial_mode = not tutorial_mode
                tutorial_playing = True
                last_tutorial_advance = now
                tutorial_cached_image = None
                tutorial_video_started = False
            elif tutorial_mode and clicked == "play_pause":
                tutorial_playing = not tutorial_playing
                last_tutorial_advance = now
            elif tutorial_mode and clicked == "next":
                tutorial_index = (tutorial_index + 1) % len(TUTORIAL_STEPS)
                last_tutorial_advance = now
                tutorial_cached_image = None
                tutorial_video_started = False
            elif tutorial_mode and clicked == "prev":
                tutorial_index = (tutorial_index - 1) % len(TUTORIAL_STEPS)
                last_tutorial_advance = now
                tutorial_cached_image = None
                tutorial_video_started = False

            if tutorial_mode and tutorial_playing and now - last_tutorial_advance > tutorial_step_ms:
                tutorial_index = (tutorial_index + 1) % len(TUTORIAL_STEPS)
                last_tutorial_advance = now
                tutorial_cached_image = None
                tutorial_video_started = False

            if tutorial_mode:
                step = TUTORIAL_STEPS[tutorial_index]
                step_gesture = step[0]
                if step_gesture in VIDEO_GESTURES:
                    if not tutorial_video_started:  # step just changed - restart the clip
                        video_caps[step_gesture].set(cv2.CAP_PROP_POS_FRAMES, 0)
                        tutorial_video_started = True
                    vframe = next_video_frame(step_gesture)
                    meme_source = vframe if vframe is not None else frame
                else:
                    if tutorial_cached_image is None:
                        tutorial_cached_image = random.choice(memes[step_gesture])
                    meme_source = tutorial_cached_image
            else:
                mirror_meme = should_mirror(current_gesture, hand_result, state)
                if current_gesture in VIDEO_GESTURES:
                    vframe = next_video_frame(current_gesture)
                    meme_source = vframe if vframe is not None else current_meme
                else:
                    meme_source = current_meme
                if mirror_meme:
                    meme_source = cv2.flip(meme_source, 1)

            # bail out if the window was closed via its titlebar X - otherwise
            # every cv2 call below (getWindowImageRect, imshow, waitKey) on a
            # destroyed window raises a low-level cv2.error.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            # read the window's actual current size every frame, so the
            # meme keeps filling the whole thing as you resize/maximize it
            # (not just the underlying webcam frame's fixed resolution).
            try:
                _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
                if win_w > 0 and win_h > 0:
                    canvas_size = (win_w, win_h)
            except cv2.error:
                pass
            win_w, win_h = canvas_size

            canvas = compose_video_call_view(meme_source, frame, win_w, win_h)
            draw_tutorial_ui(
                canvas, ui_state, tutorial_mode, tutorial_playing,
                TUTORIAL_STEPS[tutorial_index] if tutorial_mode else None,
            )
            if show_hud:
                draw_debug_hud(canvas, state, current_gesture)
            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("d"):
                show_hud = not show_hud
            elif key == ord("t"):
                tutorial_mode = not tutorial_mode
                tutorial_playing = True
                last_tutorial_advance = now
                tutorial_cached_image = None
                tutorial_video_started = False
            elif tutorial_mode and key == 32:  # spacebar
                tutorial_playing = not tutorial_playing
                last_tutorial_advance = now
            elif tutorial_mode and key == ord("n"):
                tutorial_index = (tutorial_index + 1) % len(TUTORIAL_STEPS)
                last_tutorial_advance = now
                tutorial_cached_image = None
                tutorial_video_started = False
            elif tutorial_mode and key == ord("p"):
                tutorial_index = (tutorial_index - 1) % len(TUTORIAL_STEPS)
                last_tutorial_advance = now
                tutorial_cached_image = None
                tutorial_video_started = False
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        for vcap in video_caps.values():
            vcap.release()
        flow_log.close()
        cv2.destroyAllWindows()
        hand_landmarker.close()
        face_landmarker.close()


if __name__ == "__main__":
    main()
