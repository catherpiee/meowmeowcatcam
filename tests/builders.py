"""Synthetic MediaPipe results.

MediaPipe hand/face results are plain attribute bags, so tests build them
directly instead of running a model. The geometry here is chosen so that the
values `classify_hand` derives from it land unambiguously on one side of each
threshold.
"""

import math
from types import SimpleNamespace


def _lm(x, y, z=0.0):
    return SimpleNamespace(x=float(x), y=float(y), z=float(z))


# Hand landmark indices, per MediaPipe's hand model.
WRIST = 0
THUMB_TIP = 4
FINGERS = {
    # name: (mcp, pip, tip)
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}
# x offset of each finger column, as a multiple of `scale`
FINGER_X = {"index": -0.3, "middle": 0.0, "ring": 0.3, "pinky": 0.6}


def make_hand(
    *,
    index_up=True,
    middle_up=True,
    ring_up=True,
    pinky_up=True,
    thumb_out=False,
    center=(0.5, 0.75),
    scale=0.1,
):
    """Build 21 hand landmarks.

    `scale` becomes handScale exactly: the wrist sits `scale` below the
    middle-finger MCP (landmark 9), and handScale is dist(pts[0], pts[9]).

    An extended finger is built collinear (mcp -> pip -> tip all straight up),
    so angle_deg is 0 and finger_extended's < 45 test passes. A curled finger
    folds the tip back down past the PIP, giving an angle of 180.
    """
    cx, cy = center
    pts = [_lm(cx, cy) for _ in range(21)]

    palm_center = (cx, cy)                 # landmark 9
    wrist = (cx, cy + scale)               # landmark 0, `scale` below the palm
    pts[WRIST] = _lm(*wrist)

    up = {"index": index_up, "middle": middle_up, "ring": ring_up, "pinky": pinky_up}
    seg = scale * 0.4
    for name, (mcp_i, pip_i, tip_i) in FINGERS.items():
        fx = cx + FINGER_X[name] * scale
        mcp_y = cy
        pip_y = mcp_y - seg
        pts[mcp_i] = _lm(fx, mcp_y)
        pts[pip_i] = _lm(fx, pip_y)
        # extended: keep going up (angle 0). curled: fold back down (angle 180).
        pts[tip_i] = _lm(fx, pip_y - seg if up[name] else pip_y + seg)

    # thumbOut = dist(pts[4], pts[17]) / handScale > 1.05
    pinky_mcp = pts[FINGERS["pinky"][0]]
    if thumb_out:
        pts[THUMB_TIP] = _lm(pinky_mcp.x - 2.0 * scale, cy)
    else:
        pts[THUMB_TIP] = _lm(pinky_mcp.x - 0.2 * scale, cy + 0.1 * scale)

    assert pts[9].x == palm_center[0] and pts[9].y == palm_center[1]
    return pts


def move_hand(pts, dx, dy):
    """Translate every landmark. Used to place a hand near/far from the face."""
    return [_lm(p.x + dx, p.y + dy, p.z) for p in pts]


def make_face(*, center=(0.5, 0.5), width=0.2, mouth_open=0.02):
    """Build 478 face landmarks.

    Only 13/14 (inner lips) and 234/454 (cheeks) are read by the code under
    test; the rest exist so indexing is safe.

    face_width  == `width`
    mouth_open  == `mouth_open` (it is a ratio of lip gap to face width)
    mouth_center == `center`
    """
    cx, cy = center
    pts = [_lm(cx, cy) for _ in range(478)]
    gap = mouth_open * width
    pts[13] = _lm(cx, cy - gap / 2)   # upper lip
    pts[14] = _lm(cx, cy + gap / 2)   # lower lip
    pts[234] = _lm(cx - width / 2, cy)  # right cheek
    pts[454] = _lm(cx + width / 2, cy)  # left cheek
    return pts


def yaw_matrix(deg):
    """4x4 transform for a pure rotation about Y.

    yaw_from_transform_matrix reads r[0][0], r[1][0], r[2][0] and computes
    degrees(atan2(-r[2][0], sqrt(r[0][0]**2 + r[1][0]**2))), which returns
    `deg` back for |deg| < 90.
    """
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return [
        [c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def hand_result(hands, handedness=None):
    """A stand-in for HandLandmarkerResult.

    `handedness` is a list of label strings, one per hand; it is turned into
    the nested category structure MediaPipe actually returns.
    """
    if handedness is None:
        handedness = ["Right"] * len(hands)
    return SimpleNamespace(
        hand_landmarks=list(hands),
        handedness=[[SimpleNamespace(category_name=h, score=1.0)] for h in handedness],
    )


def face_result(face=None, matrix=None):
    """A stand-in for FaceLandmarkerResult."""
    return SimpleNamespace(
        face_landmarks=[face] if face is not None else [],
        facial_transformation_matrixes=[matrix] if matrix is not None else [],
    )
