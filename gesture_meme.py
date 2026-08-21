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

The Camera window shows a live debug readout (head yaw, and optical-flow
magnitude/coherence, vs. their trigger thresholds) in the top-left corner so
side-eye and spin can both be tuned by eye - see SIDE_EYE_YAW_DEG and
SPIN_FLOW_SCORE_THRESHOLD below.

Press q or ESC to quit.
"""

import math
import random
import sys
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
            if face_result.facial_transformation_matrixes:
                yaw_deg = yaw_from_transform_matrix(face_result.facial_transformation_matrixes[0])

            self.last_face = (mouth_center, face_width, mouth_open, yaw_deg, now)
            self.last_yaw_debug = yaw_deg
        self.face_seen_this_frame = saw_face

    def decide(self, hand_result):
        now = time.time() * 1000
        face_is_fresh = self.last_face is not None and now - self.last_face[4] < FACE_STALE_MS

        # spinning in the chair beats everything else, hands included.
        if self.is_spinning(now):
            return "spinCat"

        if not hand_result.hand_landmarks:
            # no hands: side-eye is a face-only pose (head turned, no
            # particular hand shape needed).
            if face_is_fresh and abs(self.last_face[3]) > SIDE_EYE_YAW_DEG:
                return "sideEyeCat"
            return "default"

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
    ]
    for i, line in enumerate(lines):
        y = 24 + i * 22
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1, cv2.LINE_AA)


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


CAMERA_INDICES_TO_TRY = 5
CAMERA_OPEN_READS = 20
FRAME_READ_RETRIES = 30


def open_webcam():
    """Open the first camera that actually yields a frame.

    On macOS, index 0 is often a Continuity Camera (iPhone) that reports as
    opened but never produces frames. A freshly granted Camera permission
    can also make the first few reads fail on the built-in webcam.
    """
    backends = [cv2.CAP_ANY]
    if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]

    for backend in backends:
        for index in range(CAMERA_INDICES_TO_TRY):
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            for _ in range(CAMERA_OPEN_READS):
                ok, frame = cap.read()
                if ok and frame is not None and frame.size:
                    return cap
                time.sleep(0.05)
            cap.release()

    raise RuntimeError(
        "Could not open a webcam. On macOS: System Settings → Privacy & "
        "Security → Camera, enable access for Terminal, Cursor, and Python, "
        "then rerun this script."
    )


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

    # every frame's flow numbers get logged here, timestamped - so we can
    # look at exactly what a real, full-effort spin looked like afterward
    # instead of trying to read a jittery number while dizzy.
    flow_log_path = ROOT / "flow_debug_log.csv"
    flow_log = open(flow_log_path, "w", buffering=1)  # line-buffered so data survives a hard kill
    flow_log.write("t_ms,magnitude,coherence,score,fraction,peak_2s,gesture\n")

    spin_video_cap = cv2.VideoCapture(str(MEMES / GESTURE_MEMES["spinCat"][0]))
    if not spin_video_cap.isOpened():
        raise FileNotFoundError(f"missing meme file: {MEMES / GESTURE_MEMES['spinCat'][0]}")

    def next_spin_frame():
        ok, vframe = spin_video_cap.read()
        if not ok:
            spin_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, vframe = spin_video_cap.read()
        return vframe

    cap = open_webcam()

    cv2.namedWindow("Camera")
    cv2.namedWindow("Meme")
    cv2.moveWindow("Camera", 40, 80)
    cv2.moveWindow("Meme", 720, 80)

    state = GestureState()
    current_gesture = "default"
    candidate_gesture = "default"
    candidate_streak = 0
    last_non_default_at = time.time() * 1000
    current_meme = random.choice(memes["default"])
    prev_flow_gray = None

    start_time = time.time()
    missed_frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                missed_frames += 1
                if missed_frames > FRAME_READ_RETRIES:
                    break
                continue
            missed_frames = 0
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
                elif gesture == "spinCat":
                    spin_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            if gesture != "default":
                last_non_default_at = now
            elif now - last_non_default_at > DEFAULT_FALLBACK_MS and current_gesture != "default":
                current_gesture = "default"
                current_meme = random.choice(memes["default"])

            draw_landmarks(frame, hand_result)
            draw_debug_hud(frame, state, current_gesture)

            if current_gesture == "spinCat":
                vframe = next_spin_frame()
                meme_view = (
                    fit_to_height(vframe, frame.shape[0])
                    if vframe is not None
                    else fit_to_height(current_meme, frame.shape[0])
                )
            else:
                meme_view = fit_to_height(current_meme, frame.shape[0])
            cv2.imshow("Camera", frame)
            cv2.imshow("Meme", meme_view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        spin_video_cap.release()
        flow_log.close()
        cv2.destroyAllWindows()
        hand_landmarker.close()
        face_landmarker.close()


if __name__ == "__main__":
    main()
