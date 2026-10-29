#!/bin/sh
# Starts a worker on macOS or Linux, setting everything up the first time, with nothing to install
# first. Straight from the server, which fills in its address and key:
#
#   curl -fsSL https://lyrics.example.com/workers/start/<WORKER_KEY> | sh
#
# or from a copy of this folder: ./run.sh --server https://lyrics.example.com --key <WORKER_KEY>
# Anything after it goes to worker.py (from the server: ... | sh -s -- --name gaming-pc).
#
# It all goes in one folder: this one, or from the server ~/.jamtube-lyrics-worker
# (JAMTUBE_WORKER_DIR). uv, the Python it brings, what the worker needs and the model. Deleting
# the folder removes the worker.

# The server fills these in when it hands this script out.
SERVER=''
KEY=''

PYTHON_VERSION=${PYTHON_VERSION:-3.12} # the Python vocal removal runs on (3.10 to 3.13)
UV_INSTALLER=${UV_INSTALLER:-https://astral.sh/uv/install.sh}

# All in a function, so it's all been read before it starts (it's read from a pipe).
main() {
  set -e
  if [ -n "$SERVER" ]; then
    DIR=${JAMTUBE_WORKER_DIR:-$HOME/.jamtube-lyrics-worker}
    mkdir -p "$DIR"
    echo "Getting the worker from $SERVER"
    for file in worker.py requirements-cpu.txt requirements-gpu.txt; do
      curl -fsSL -H "Authorization: Bearer $KEY" "$SERVER/workers/files/$file" -o "$DIR/$file"
    done
    set -- --server "$SERVER" --key "$KEY" "$@"
  else
    DIR=$(cd "$(dirname "$0")" && pwd)
  fi
  cd "$DIR"

  # uv: it brings the right Python and installs what the worker needs, all in this folder.
  export UV_CACHE_DIR="$DIR/.uv/cache" UV_PYTHON_INSTALL_DIR="$DIR/.uv/python"
  UV="$DIR/.uv/bin/uv"
  if [ ! -x "$UV" ]; then
    echo "Getting uv (it brings Python)"
    curl -LsSf "$UV_INSTALLER" | env UV_INSTALL_DIR="$DIR/.uv/bin" UV_NO_MODIFY_PATH=1 sh >/dev/null
  fi
  if [ ! -x .venv/bin/python ]; then
    "$UV" venv --quiet --python "$PYTHON_VERSION" .venv
  fi

  # The GPU build with an NVIDIA GPU. .venv/installed is what was installed last.
  if command -v nvidia-smi >/dev/null 2>&1; then KIND=gpu; else KIND=cpu; fi
  if ! cmp -s "requirements-$KIND.txt" .venv/installed; then
    echo "Installing what it needs ($KIND): the first time only, a few minutes"
    "$UV" pip install --python .venv/bin/python --index-strategy unsafe-best-match -r "requirements-$KIND.txt"
    cp "requirements-$KIND.txt" .venv/installed
  fi

  exec .venv/bin/python worker.py "$@"
}

main "$@"
