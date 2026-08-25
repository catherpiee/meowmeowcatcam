# Meowmeow cat cam meme detector

Point your webcam at yourself, make a face/hand gesture, get a cat meme back in real time. Runs either as a desktop app (OpenCV windows) or entirely in the browser (MediaPipe WASM, no install).

Two windows/panes side by side: 
- **Camera** — your webcam feed with hand landmarks drawn on top, plus a live debug readout in the corner
- **Meme** — the meme matching whatever gesture you're currently making

## Gestures

Checked in this order — when a pose could match more than one, the earlier one wins.

| # | Gesture | How to trigger |
|---|---|---|
| 1 | Spinny OIIAI cat | You spin!!!! (beats everything else, hands included) |
| 2 | Muehehe | Both hands up, index fingers only, tips touching |
| 3 | Devo cat | Both hands up, above the top of your head |
| 4 | Crash out cord chewing kitty | Both hands up beside your face to hold yummy electrical cable |
| 5 | Thumbs up cat | One hand, thumb sticking out, other four fingers curled |
| 6 | I will punch you | One hand, all four fingers curled, thumb tucked in |
| 7 | EHHEHEEEHEEEE | Thumb + pinky out, rockstar cat |
| 8 | Blep cat | One hand, all four fingers up, thumb tucked in ("four" sign) |
| 9 | Scratch-n-sniff cat | One hand, three fingers up (index + middle + ring) |
| 10 | Shocked cat | One hand, two fingers up (index + middle, peace sign) |
| 11 | Thinking cat | Index finger only, tip propped near your chin |
| 12 | Shhh silenced cat | Index finger only, tip resting on your mouth |
| 13 | Erm ackshuALLY! cat / beluga smile cat | Index finger only, held away from your face |
| 14 | Shocked/kidnapped cat | Hand cover mouth |
| 15 | gGIMME MONIE!! | One open palm, all fingers extended (thumb included), away from your face |
| 16 | Side eye cat | Turn your head 15°+ either way (real head-pose yaw) |
| 17 | Pokercat | Default |


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
```

Useful for tuning the detection thresholds at the top of `gesture_meme.py` / `app.js` if a gesture is triggering too easily or not easily enough for your setup/lighting.

## Project layout

```
gesture_meme.py   desktop version (OpenCV + MediaPipe Python tasks API)
app.js            browser version (MediaPipe tasks-vision WASM)
index.html        browser UI shell
memes/            meme images (+ one video, unused for now)
models/           MediaPipe .task model files used by the desktop version
requirements.txt  Python dependencies
```
