#!/bin/bash
cd "$(dirname "$0")"

if [ ! -f .venv/bin/python3 ]; then
    echo "First run — setting up (this takes a minute)..."
    python3 -m venv .venv
    .venv/bin/pip install --quiet --upgrade pip
    .venv/bin/pip install --quiet -r requirements.txt
fi

.venv/bin/python3 gesture_meme.py
