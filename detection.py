"""Pure hand/face shape math and gesture classification.

This module deliberately depends only on ``numpy`` and ``math`` -- no OpenCV,
no MediaPipe camera objects -- so the classifier can be unit-tested without a
webcam. ``gesture_meme.py`` owns the camera, optical flow, rendering and the
main loop, and imports the functions here.

Landmarks are any objects exposing ``.x``, ``.y`` and ``.z`` (MediaPipe's
NormalizedLandmark, or a plain namedtuple in tests). Coordinates are
image-normalised: x right, y down, z depth.
"""

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

# ---- low-level geometry constants (not runtime-calibrated) ----------------
FINGER_STRAIGHT_ANGLE_DEG = 45.0  # a finger straighter than this counts as "not bent"
THUMB_STRAIGHT_ANGLE_DEG = 50.0   # thumbs bend at a wider angle than fingers
THUMB_EXTENDED_MIN_SPREAD = 0.45  # thumb tip distance from the index MCP, over hand scale


@dataclass
class Tunables:
    """Setup-dependent decision thresholds, gathered in one place.

    Seeded from values tuned by eye against real sessions. These are the knobs
    the live calibration keys nudge; the low-level geometry constants above are
    not calibrated. Kept in-source (no config file) by design decision.
    """

    side_eye_yaw_deg: float = 15.0
    # spin (optical-flow) detection
    spin_mag_threshold: float = 0.8
    spin_fraction_required: float = 0.55
    spin_fraction_window_ms: float = 2200.0
    # hand-covering-face proximity (palm-to-mouth, over face width)
    hand_cover_face_dist_face_lost: float = 1.3
    hand_cover_face_dist_face_seen: float = 0.7
    # other proximity ratios
    shhh_mouth_dist: float = 0.55        # index tip to mouth, over face width
    two_fingers_tip_gap: float = 1.4     # index-tip gap, over hand scale
    near_face_palm_dist: float = 2.2     # palm to mouth, over face width
    head_top_offset: float = 1.1         # head-top height above mouth, over face width
    # detector + loop
    min_hand_confidence: float = 0.6
    min_face_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    min_handedness_score: float = 0.6    # drop hands the detector is unsure are hands
    stable_frames: int = 5               # sliding majority-vote window length
    flow_every_n: int = 2                # compute optical flow every Nth frame

    def fields_for_calibration(self):
        """(label, attr) pairs exposed to the live calibration keys."""
        return [
            ("side_eye_yaw_deg", "side_eye_yaw_deg"),
            ("spin_mag_threshold", "spin_mag_threshold"),
            ("spin_fraction_required", "spin_fraction_required"),
            ("hand_cover(face lost)", "hand_cover_face_dist_face_lost"),
            ("hand_cover(face seen)", "hand_cover_face_dist_face_seen"),
            ("shhh_mouth_dist", "shhh_mouth_dist"),
            ("two_fingers_tip_gap", "two_fingers_tip_gap"),
            ("near_face_palm_dist", "near_face_palm_dist"),
            ("head_top_offset", "head_top_offset"),
        ]


# ---- geometry helpers -----------------------------------------------------
def p3(lm):
    return np.array([lm.x, lm.y, lm.z], dtype=float)


def dist(a, b):
    """3D distance. Use only when both points come from the same model."""
    return float(np.linalg.norm(a - b))


def dist2d(a, b):
    """Image-plane (x, y) distance. Safe across the hand and face models,
    whose z axes are not comparable."""
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def angle_deg(v1, v2):
    m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if m1 < 1e-9 or m2 < 1e-9:
        return 180.0
    cos_a = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def yaw_from_transform_matrix(matrix):
    """Head left/right turn angle (yaw, degrees) from MediaPipe's facial
    transformation matrix -- its own head-pose estimate, far more robust than
    inferring turn from landmark distances."""
    r = np.asarray(matrix)[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy < 1e-6:
        return 0.0
    return math.degrees(math.atan2(-r[2, 0], sy))


# ---- finger / thumb / palm shape ------------------------------------------
def finger_extended(pts, mcp, pip, tip):
    """A finger is extended when it is both roughly straight AND reaching away
    from the wrist. The wrist-distance test rejects a finger that is straight
    but folded back toward the palm -- an angle-only test called that extended."""
    if angle_deg(pts[pip] - pts[mcp], pts[tip] - pts[pip]) >= FINGER_STRAIGHT_ANGLE_DEG:
        return False
    wrist = pts[0]
    return dist2d(pts[tip], wrist) > dist2d(pts[pip], wrist)


def thumb_extended(pts):
    """The thumb (indices 1-4) is extended when it is straight at the IP joint
    and its tip is spread away from the index MCP, rather than tucked across
    the palm. Handled separately because the thumb bends differently."""
    if angle_deg(pts[3] - pts[2], pts[4] - pts[3]) >= THUMB_STRAIGHT_ANGLE_DEG:
        return False
    hand_scale = dist2d(pts[0], pts[9]) or 1e-6
    return dist2d(pts[4], pts[5]) / hand_scale > THUMB_EXTENDED_MIN_SPREAD


def palm_normal(pts):
    """Unit normal of the palm plane (wrist, index-MCP, pinky-MCP)."""
    n = np.cross(pts[5] - pts[0], pts[17] - pts[0])
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-9 else n


def palm_facing_camera(pts, handedness_label=None):
    """Whether the palm faces the camera. The sign of the palm normal's z
    flips between a left and right hand, so handedness selects it. The raw
    normal-z is surfaced in the HUD so this can be tuned on-camera."""
    nz = float(palm_normal(pts)[2])
    if handedness_label == "Left":
        return nz > 0.0
    return nz < 0.0


def _handedness_label(handedness):
    if handedness is None:
        return None, None
    if isinstance(handedness, str):
        return handedness, None
    return getattr(handedness, "category_name", None), getattr(handedness, "score", None)


def classify_hand(landmarks, handedness=None):
    pts = [p3(lm) for lm in landmarks]
    hand_scale = dist2d(pts[0], pts[9]) or 1e-6

    index_up = finger_extended(pts, 5, 6, 8)
    middle_up = finger_extended(pts, 9, 10, 12)
    ring_up = finger_extended(pts, 13, 14, 16)
    pinky_up = finger_extended(pts, 17, 18, 20)
    thumb_up = thumb_extended(pts)

    four = (index_up, middle_up, ring_up, pinky_up)
    label, score = _handedness_label(handedness)

    return {
        "indexUp": index_up,
        "middleUp": middle_up,
        "ringUp": ring_up,
        "pinkyUp": pinky_up,
        "thumbExtended": thumb_up,
        "fingersExtended": sum(1 for v in four if v),
        "curledCount": sum(1 for v in four if not v),
        "handScale": hand_scale,
        "indexTip": pts[8],
        "wrist": pts[0],
        "palmCenter": pts[9],
        "palmNormalZ": float(palm_normal(pts)[2]),
        "palmFacingCamera": palm_facing_camera(pts, label),
        "handedness": label,
        "handednessScore": score,
    }


def is_pointing(h):
    return h["indexUp"] and not h["middleUp"] and not h["ringUp"] and not h["pinkyUp"]


def stable_gesture(recent, current):
    """Majority-vote flicker smoother. ``recent`` is a bounded sequence of the
    most recent per-frame gesture guesses. The displayed gesture only changes
    away from ``current`` once some gesture holds a strict majority of the
    window, so a single stray frame can't flip it."""
    recent = list(recent)
    if not recent:
        return current
    winner, count = Counter(recent).most_common(1)[0]
    if count >= len(recent) // 2 + 1:
        return winner
    return current
