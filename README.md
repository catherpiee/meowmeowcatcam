# Meowmeow cat cam meme detector

Point your webcam at yourself, make a face/hand gesture, get a cat meme back in real time. Runs either as a desktop app (OpenCV windows) or entirely in the browser (MediaPipe WASM, no install).

Two windows/panes side by side: 
- **Camera** — your webcam feed with hand landmarks drawn on top, plus a live debug readout in the corner
- **Meme** — the meme matching whatever gesture you're currently making

## Gestures

Checked in this order — when a pose could match more than one, the earlier one wins.

| # | Gesture | How to trigger |
|---|---|---|
| 1 | Muehehe | Both hands up, index fingers only, tips touching |
| 2 | Devo cat | Both hands up, above the top of your head |
| 3 | Crash out cord chewing kitty | Both hands up beside your face to hold yummy electrical cable |
| 4 | I will punch you | One hand, all four fingers curled |
| 5 | EHHEHEEEHEEEE | Thumb + pinky out, rockstar cat |
| 6 | Shhh silenced cat | Index finger only, tip resting on your mouth |
| 7 | Erm ackshuALLY! cat | Index finger only, held away from your face |
| 8 | Shocked/kidnapped cat | Hand cover mouth |
| 9 | gGIMME MONIE!! | One open palm, all fingers extended, away from your face |
| 10 | Side eye cat | Turn your head 15°+ either way (real head-pose yaw) |
| 11 | Pokercat | Default |
| 12 | Spinny OIIAI cat | You spin!!!! |


Meme images live in `memes/`. A couple of gestures pick randomly between multiple images.

## Running it — desktop (Python)

Requires Python 3 and a webcam.

Easiest way: just double-click **`Launch Gesture Meme.command`**. First run takes a minute to set itself up (installs everything automatically), then launches straight away. Every run after that is instant.

**First time opening it:** macOS will warn "cannot be opened because it is from an unidentified developer" — this is normal for any downloaded script, not specific to this one. Right-click the file → **Open** → click **Open** in the dialog that appears. You only need to do this once.

Or manually, if you prefer Terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 gesture_meme.py
```

Press `q` or `Esc` in the Camera window to quit.

## Running it — browser

No install needed, but the webcam API requires serving over HTTP (opening `index.html` directly as a `file://` URL will not get camera permission). From this folder:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000` and allow camera access. Models load from Google's hosted MediaPipe CDN at runtime, so nothing local is needed for the browser version.

## Live debug HUD

The Camera window always shows a small readout in the top-left corner:

```
gesture: sideEyeCat
yaw: +18.4 deg  (side-eye thr +/-15.0)
...
calib [1/9] side_eye_yaw_deg: 15.0   ([ ] pick  - = nudge  p print)
```

Useful for seeing why a gesture is or isn't triggering for your setup/lighting.

### Live calibration keys

The detection thresholds are collected in the `Tunables` dataclass in
`detection.py`. You can adjust them **while the app runs**, without editing code:

| Key | Action |
|---|---|
| `[` / `]` | pick which threshold the HUD's `calib` line is showing |
| `-` / `=` | nudge that threshold down / up (5% steps) |
| `p` | print all current thresholds to the terminal |

Nothing is written to disk — once a value feels right, press `p` and paste it
into `Tunables` in `detection.py` to make it the new default.

## Running the tests

The hand/gesture classification logic lives in `detection.py` (pure geometry,
no webcam) and is covered by `test_detection.py`:

```bash
python3 test_detection.py      # no extra dependencies
# or, with pytest installed:
pytest test_detection.py
```

## Project layout

```
gesture_meme.py     desktop version: camera, optical flow, rendering, main loop
detection.py        pure hand/gesture classification + thresholds (webcam-free)
test_detection.py   unit tests for detection.py
app.js              browser version (MediaPipe tasks-vision WASM)
index.html          browser UI shell
memes/              meme images (+ one video, unused for now)
models/             MediaPipe .task model files used by the desktop version
requirements.txt    Python dependencies
```
