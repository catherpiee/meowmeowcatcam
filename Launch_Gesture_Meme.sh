#!/bin/bash
cd "$(dirname "$0")"

if [ ! -f .venv/bin/python3 ]; then
    echo "First run — setting up (this takes a minute)..."
    python3 -m venv .venv
fi

.venv/bin/python3 -m pip install --quiet --upgrade pip
.venv/bin/python3 -m pip install --quiet --upgrade -r requirements.txt
.venv/bin/python3 gesture_meme.py
