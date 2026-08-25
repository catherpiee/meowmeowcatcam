# Manual test checklist

No automated test covers the camera, MediaPipe inference, or window
rendering. Walk this list on a real webcam after any change to the rules.

Run: `.venv/bin/python3 gesture_meme.py`

Watch the HUD line `gesture: X   candidate: Y xN`. `candidate` is what the
classifier sees right now; `gesture` is what is on screen. If `candidate`
flickers between values while you hold still, that pose is unstable and the
thresholds need work — note which values it alternates between.

| # | Pose | Expected gesture | Pass? |
|---|---|---|---|
| 1 | Face the camera, hands down | `default` | |
| 2 | Turn your head ~30 degrees, hands down | `sideEyeCat` | |
| 3 | Both hands flat on top of your head | `twoHandsOnHead` | |
| 4 | Both hands beside your face, below head height | `crashOutCat` | |
| 5 | Both index fingers pointing, tips nearly touching | `twoFingersTogether` | |
| 6 | One closed fist, held away from your face | `fist` | |
| 7 | Thumb and pinky out, other fingers curled | `rockstar` | |
| 8 | Index finger vertical against your lips | `shhh` | |
| 9 | Index finger pointing away | `oneFingerUp` | |
| 10 | Open palm covering your face | `handCoverFace` | |
| 11 | Open palm held out to the side | `handStretchedOut` | |
| 12 | Spin your chair a full turn | `spinCat` | |

Stability checks:

| # | Check | Expected | Pass? |
|---|---|---|---|
| 13 | Hold pose 6 with your other hand also visible and resting | stays `fist`, does not flicker | |
| 14 | Repeat 13 with the hands swapped left/right | same result as 13 | |
| 15 | Switch quickly between poses 6 and 7 | each meme visibly holds ~0.7s, no strobing | |
| 16 | Drop both hands from any pose | returns to `default` within ~1s | |

If any row fails, record the HUD's `candidate` readout and the value in
`flow_debug_log.csv` at that moment before changing a threshold.
