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
side-eye and spin can both be tuned by eye. All detection thresholds live in
detection.Tunables and can be nudged live with the calibration keys
([ ] to pick, - = to nudge, p to print) - see the README.

The hand/gesture classification itself lives in detection.py (pure geometry,
no camera) so it can be unit-tested; this file owns the camera, optical flow,
rendering and the main loop.

Press q or ESC to quit.
"""

import random
import sys
import time
from collections import deque
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

from detection import (
    Tunables,
    classify_hand,
    dist,
    dist2d,
    is_pointing,
    p3,
    stable_gesture,
    yaw_from_transform_matrix,
)

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

DEFAULT_FALLBACK_MS = 600
FACE_STALE_MS = 1200

# All setup-dependent decision thresholds (side-eye yaw, spin fraction,
# hand-to-face proximity, detector confidence, etc.) now live in
# detection.Tunables, so they can be adjusted live by the calibration keys and
# tested independently. See that dataclass for the documented defaults; the
# comments about how the spin fraction test was tuned moved there too.

# Optical-flow computation constants (fixed pipeline params, not calibrated):
# the frame is downsized to this size for speed before dense flow is computed.
SPIN_FLOW_WIDTH = 160
SPIN_FLOW_HEIGHT = 90
SPIN_FLOW_NOISE_FLOOR_PX = 0.4  # per-pixel motion below this is treated as noise, not real motion
SPIN_FLOW_MIN_MOVING_FRACTION = 0.15  # need at least this much of the frame moving to trust coherence at all
SPIN_FLOW_PEAK_HOLD_MS = 2000  # trailing window for the readable peak-score HUD display

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# Geometry helpers and hand classification now live in detection.py so they can
# be unit-tested without a webcam; they are imported at the top of this file.


def downsize_gray(frame):
    """Cheap per-frame step: grayscale + shrink for optical flow. Kept separate
    from the (expensive) flow computation so callers can run the shrink every
    frame but the flow only every Nth frame (see Tunables.flow_every_n)."""
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (SPIN_FLOW_WIDTH, SPIN_FLOW_HEIGHT))


def flow_signal(prev_small_gray, small_gray):
    """Dense optical flow between two consecutive downsized frames, reduced to
    (magnitude, coherence): how much of the frame moved on the horizontal axis,
    and what fraction of that motion agreed on one direction. Always measured
    over a single frame step, so its magnitude scale is independent of how often
    it is called."""
    flow = cv2.calcOpticalFlowFarneback(prev_small_gray, small_gray, None, 0.5, 2, 15, 2, 5, 1.2, 0)
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

    return magnitude, coherence


class GestureState:
    def __init__(self, tun=None):
        self.tun = tun or Tunables()
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
        self.flow_history = [(t, m) for t, m in self.flow_history if now - t < self.tun.spin_fraction_window_ms]

        self.flow_peak_history.append((now, score))
        self.flow_peak_history = [
            (t, s) for t, s in self.flow_peak_history if now - t < SPIN_FLOW_PEAK_HOLD_MS
        ]

        self.last_flow_magnitude_debug = magnitude
        self.last_flow_coherence_debug = coherence
        self.last_flow_score_debug = score
        self.last_flow_peak_debug = max((s for _, s in self.flow_peak_history), default=0.0)
        elevated = sum(1 for _, m in self.flow_history if m > self.tun.spin_mag_threshold)
        self.last_flow_fraction_debug = elevated / len(self.flow_history) if self.flow_history else 0.0

    def is_spinning(self, now):
        self.flow_history = [(t, m) for t, m in self.flow_history if now - t < self.tun.spin_fraction_window_ms]
        if not self.flow_history:
            return False
        elevated = sum(1 for _, m in self.flow_history if m > self.tun.spin_mag_threshold)
        fraction = elevated / len(self.flow_history)
        return fraction > self.tun.spin_fraction_required

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

    def _hands_from(self, hand_result):
        """Classify each detected hand, attaching its MediaPipe handedness,
        dropping low-confidence detections, and ordering the hands left-to-right
        so two-hand logic is deterministic regardless of detection order."""
        landmarks = hand_result.hand_landmarks
        handedness = getattr(hand_result, "handedness", None) or []
        hands = []
        for i, lm in enumerate(landmarks):
            entry = handedness[i] if i < len(handedness) else None
            if isinstance(entry, (list, tuple)):
                entry = entry[0] if entry else None
            h = classify_hand(lm, entry)
            score = h["handednessScore"]
            if score is not None and score < self.tun.min_handedness_score:
                continue
            hands.append(h)
        hands.sort(key=lambda h: float(h["palmCenter"][0]))
        return hands

    def _hand_gesture(self, hands, face_is_fresh):
        """The gesture implied purely by the hand shapes, or None if the hands
        aren't making any specific gesture. Kept separate from decide() so that
        spin can be gated on 'no specific hand gesture present'."""
        if not hands:
            return None

        if len(hands) >= 2:
            a, b = hands[0], hands[1]
            if is_pointing(a) and is_pointing(b):
                avg_scale = (a["handScale"] + b["handScale"]) / 2 or 1e-6
                tip_gap = dist2d(a["indexTip"], b["indexTip"]) / avg_scale
                if tip_gap < self.tun.two_fingers_tip_gap:
                    return "twoFingersTogether"

            if face_is_fresh:
                mouth_center, face_width, _, _, _ = self.last_face
                near_face = all(
                    dist2d(h["palmCenter"], mouth_center) / face_width < self.tun.near_face_palm_dist
                    for h in hands
                )
                if near_face:
                    head_top_y = mouth_center[1] - face_width * self.tun.head_top_offset
                    if all(h["palmCenter"][1] < head_top_y for h in hands):
                        return "twoHandsOnHead"
                    return "crashOutCat"

        h = hands[0]

        # thumb tucked in confirms a real fist rather than a partly-open hand.
        if h["curledCount"] == 4 and not h["thumbExtended"]:
            return "fist"

        if (
            h["thumbExtended"] and h["pinkyUp"]
            and not h["indexUp"] and not h["middleUp"] and not h["ringUp"]
        ):
            return "rockstar"

        # shhh / one-finger-up: a single extended index finger is a very
        # specific shape (shhh in particular = fingertip right on the mouth), so
        # it must be checked before the broader hand-covering-face test below.
        if h["indexUp"] and not h["middleUp"] and not h["ringUp"] and not h["pinkyUp"]:
            if face_is_fresh:
                mouth_center, face_width, _, _, _ = self.last_face
                if dist2d(h["indexTip"], mouth_center) / face_width < self.tun.shhh_mouth_dist:
                    return "shhh"
            return "oneFingerUp"

        # hand covering face: the one hand we see sits roughly where the face
        # last was. Wider tolerance if the face detector has fully lost the face
        # (strong evidence of real occlusion); tighter if it's still tracking.
        if face_is_fresh:
            mouth_center, face_width, _, _, _ = self.last_face
            d = dist2d(h["palmCenter"], mouth_center) / face_width
            threshold = (
                self.tun.hand_cover_face_dist_face_lost
                if not self.face_seen_this_frame
                else self.tun.hand_cover_face_dist_face_seen
            )
            if d < threshold:
                return "handCoverFace"

        # open palm held out toward the camera, not near the face
        if h["fingersExtended"] == 4 and h["thumbExtended"] and h["palmFacingCamera"]:
            return "handStretchedOut"

        return None

    def decide(self, hand_result):
        now = time.time() * 1000
        face_is_fresh = self.last_face is not None and now - self.last_face[4] < FACE_STALE_MS

        hands = self._hands_from(hand_result) if hand_result.hand_landmarks else []
        hand_gesture = self._hand_gesture(hands, face_is_fresh)

        # A specific hand gesture wins over spin: fast hand-waving produces the
        # same horizontal optical flow as a chair spin, so spin only takes over
        # when the hands aren't clearly forming a gesture.
        if hand_gesture is not None:
            return hand_gesture

        if self.is_spinning(now):
            return "spinCat"

        # face-only poses (no gesturing hands): a turned head is a side-eye.
        if face_is_fresh and abs(self.last_face[3]) > self.tun.side_eye_yaw_deg:
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


def draw_debug_hud(frame, state, gesture, calib=None):
    tun = state.tun
    lines = [
        f"gesture: {gesture}",
        f"yaw: {state.last_yaw_debug:+.1f} deg  (side-eye thr +/-{tun.side_eye_yaw_deg:.1f})",
        f"flow mag: {state.last_flow_magnitude_debug:.2f}  (thr {tun.spin_mag_threshold:.2f})",
        f"spin fraction ({tun.spin_fraction_window_ms / 1000:.1f}s window): "
        f"{state.last_flow_fraction_debug:.2f}  (thr {tun.spin_fraction_required:.2f})",
        f"peak score (last 2s): {state.last_flow_peak_debug:.2f}  <- read this AFTER you stop spinning",
    ]
    if calib is not None:
        lines.append(
            f"calib [{calib.i + 1}/{len(calib.fields)}] {calib.label}: {calib.value}"
            f"   ({Calibration.KEYS_HELP})"
        )
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


class Calibration:
    """Live threshold tuning from the keyboard, so you can dial detection in on
    your own setup without editing code and restarting. Nudges the running
    Tunables in place; nothing is written to disk. Press 'p' to print the
    current values so you can paste the good ones back into Tunables."""

    KEYS_HELP = "[ ] pick  - = nudge  p print"

    def __init__(self, tun):
        self.tun = tun
        self.fields = tun.fields_for_calibration()
        self.i = 0

    @property
    def label(self):
        return self.fields[self.i][0]

    @property
    def value(self):
        return getattr(self.tun, self.fields[self.i][1])

    def _nudge(self, direction):
        attr = self.fields[self.i][1]
        v = getattr(self.tun, attr)
        step = max(abs(v) * 0.05, 0.05)  # 5% of the value, with a sensible floor
        setattr(self.tun, attr, round(v + direction * step, 4))

    def dump(self):
        print("\n--- calibrated thresholds (paste into detection.Tunables) ---")
        for _, attr in self.fields:
            print(f"    {attr}: float = {getattr(self.tun, attr)}")
        print("-------------------------------------------------------------")

    def handle_key(self, key):
        if key == ord("["):
            self.i = (self.i - 1) % len(self.fields)
        elif key == ord("]"):
            self.i = (self.i + 1) % len(self.fields)
        elif key == ord("-"):
            self._nudge(-1)
        elif key in (ord("="), ord("+")):
            self._nudge(1)
        elif key == ord("p"):
            self.dump()


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
    tun = Tunables()

    hand_landmarker = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "hand_landmarker.task")),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=tun.min_hand_confidence,
            min_hand_presence_confidence=tun.min_hand_confidence,
            min_tracking_confidence=tun.min_tracking_confidence,
        )
    )
    face_landmarker = FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "face_landmarker.task")),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=tun.min_face_confidence,
            min_face_presence_confidence=tun.min_face_confidence,
            min_tracking_confidence=tun.min_tracking_confidence,
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

    state = GestureState(tun)
    calib = Calibration(tun)
    current_gesture = "default"
    recent_gestures = deque(maxlen=tun.stable_frames)  # sliding window for the majority vote
    last_non_default_at = time.time() * 1000
    current_meme = random.choice(memes["default"])
    prev_flow_gray = None
    frame_idx = 0

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

            # Optical flow is the expensive step, so run it only every Nth frame.
            # The shrink is cheap and happens every frame, so each flow is still
            # measured between two consecutive frames (its magnitude scale is
            # unchanged) - we just compute it less often.
            frame_idx += 1
            small_gray = downsize_gray(frame)
            if prev_flow_gray is not None and frame_idx % tun.flow_every_n == 0:
                magnitude, coherence = flow_signal(prev_flow_gray, small_gray)
                state.update_flow(magnitude, coherence)
                flow_log.write(
                    f"{time.time() * 1000:.0f},{magnitude:.4f},{coherence:.4f},"
                    f"{state.last_flow_score_debug:.4f},{state.last_flow_fraction_debug:.4f},"
                    f"{state.last_flow_peak_debug:.4f},{current_gesture}\n"
                )
            prev_flow_gray = small_gray

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.time() - start_time) * 1000)

            hand_result = hand_landmarker.detect_for_video(mp_image, ts_ms)
            face_result = face_landmarker.detect_for_video(mp_image, ts_ms)
            state.update_face(face_result)

            gesture = state.decide(hand_result)

            now = time.time() * 1000
            # Anti-flicker: switch to the gesture that wins a majority of the
            # last few frames, rather than trusting any single frame. This
            # settles the churn between the face-adjacent poses (crash-out /
            # hands-on-head / hand-cover-face) that share overlapping shapes.
            recent_gestures.append(gesture)
            stable = stable_gesture(recent_gestures, current_gesture)
            if stable != current_gesture:
                current_gesture = stable
                if stable not in VIDEO_GESTURES:
                    current_meme = random.choice(memes[stable])
                elif stable == "spinCat":
                    spin_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            if gesture != "default":
                last_non_default_at = now
            elif now - last_non_default_at > DEFAULT_FALLBACK_MS and current_gesture != "default":
                current_gesture = "default"
                current_meme = random.choice(memes["default"])

            draw_landmarks(frame, hand_result)
            draw_debug_hud(frame, state, current_gesture, calib)

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
            if key != 255:  # 255 == no key this frame
                calib.handle_key(key)
    finally:
        cap.release()
        spin_video_cap.release()
        flow_log.close()
        cv2.destroyAllWindows()
        hand_landmarker.close()
        face_landmarker.close()


if __name__ == "__main__":
    main()
