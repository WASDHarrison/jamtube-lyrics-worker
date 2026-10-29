#!/bin/sh
# Runs a worker on macOS or Linux, setting it up the first time (a Python environment in .venv,
# with the GPU build when there's an NVIDIA GPU). Anything after it goes to worker.py:
#
#   ./run.sh --server https://lyrics.example.com --key <WORKER_KEY>
#
# (or set JAMTUBE_LYRICS_URL and WORKER_KEY). PYTHON picks the Python to use.
set -e
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  # A Python the vocal removal works with (3.10 to 3.13), newest first.
  for candidate in "$PYTHON" python3.13 python3.12 python3.11 python3.10 python3; do
    if [ -n "$candidate" ] && command -v "$candidate" >/dev/null 2>&1 \
      && "$candidate" -c 'import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 13))'; then
      PYTHON=$candidate
      break
    fi
  done
  if [ -z "$PYTHON" ] || ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Needs Python 3.10 to 3.13 (macOS: brew install python@3.12; Linux: your package manager)." >&2
    exit 1
  fi
  "$PYTHON" -m venv .venv
fi

if command -v nvidia-smi >/dev/null 2>&1; then KIND=gpu; else KIND=cpu; fi
if [ ! -f .venv/installed ] || [ "$(cat .venv/installed)" != "$KIND" ] || [ "requirements-$KIND.txt" -nt .venv/installed ]; then
  echo "Installing what it needs ($KIND), the first time only…"
  .venv/bin/python -m pip install --quiet --upgrade pip
  .venv/bin/python -m pip install --quiet -r "requirements-$KIND.txt"
  echo "$KIND" > .venv/installed
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Needs ffmpeg (macOS: brew install ffmpeg; Linux: your package manager, e.g. apt install ffmpeg)." >&2
  exit 1
fi

exec .venv/bin/python worker.py "$@"
