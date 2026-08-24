#!/bin/bash
cd "$(dirname "$0")"

# Create the venv if it doesn't exist yet.
if [ ! -f .venv/bin/python3 ]; then
    echo "First run — creating virtual environment..."
    if ! python3 -m venv .venv; then
        echo "Could not create the virtual environment. Is python3 installed?" >&2
        exit 1
    fi
fi

# Keep pip current, but never let that gate the dependency sync below: the two
# are independent, and a failed self-upgrade (offline, proxy, locked-down box)
# must not stop requirements.txt from being applied.
.venv/bin/pip install --quiet --upgrade pip || \
    echo "Warning: could not upgrade pip; continuing." >&2

# Sync dependencies on every launch. Checking "can we import it?" was not enough:
# an already-installed but broken version (e.g. mediapipe 1.0.1, which imports
# fine and then aborts at runtime) passed the check, so the pins in
# requirements.txt never got applied. pip is a fast no-op when nothing changes.
echo "Checking dependencies..."
if ! .venv/bin/pip install --quiet -r requirements.txt; then
    # Installing failed — that's only fatal if the app can't actually run
    # (e.g. first run, or offline with nothing installed yet). Check the pinned
    # mediapipe version explicitly, not just importability: the version this pin
    # exists to keep out imports cleanly and only dies once a graph is built.
    if ! .venv/bin/python3 -c \
        "import cv2, numpy, mediapipe; assert mediapipe.__version__ == '1.0.0'" 2>/dev/null; then
        echo "Dependency installation failed. See the error above." >&2
        exit 1
    fi
    echo "Warning: could not check for updates (offline?). Using what's installed." >&2
fi

.venv/bin/python3 gesture_meme.py
