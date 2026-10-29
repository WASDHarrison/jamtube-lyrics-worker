# JamTube Lyrics Worker

Takes the vocals out of songs for a [JamTube Lyrics](https://github.com/WASDHarrison/jamtube-lyrics) server, on another computer, ideally one with a GPU. Removing the vocals is the slow part of making a karaoke song: a couple of minutes on a small server's CPU, seconds on a GPU.

Run as many workers as you like, wherever you like: your laptop, a gaming PC, a cloud GPU for the night. Each one connects to the server, asks for a song, takes its vocals out, sends the result back and asks for the next. The server gives each song to a free worker, soonest-needed first, and does it itself when none is free. If a worker fails, is stopped or goes quiet (switched off, lost its connection), its song goes to another worker, or back to the server. So a worker can come and go at any time, and nothing breaks.

A worker keeps nothing: each song is downloaded into the worker's own temporary folder and deleted once it's sent back, and the folder goes when the worker stops. (A worker killed outright, without the chance to stop, leaves its folder to the system, which clears temporary files; a cloud machine or container takes it with it.) Only the model (about 60 MB) is kept, so it's downloaded once. It removes the vocals exactly as the server would, with the same separator, model and settings, which the server sends with each song. The result is the same, sample for sample.

## On the server

Set `WORKER_KEY` in JamTube Lyrics' `.env` to a long random secret (`openssl rand -hex 32`), and restart it. Workers then connect with that key. The server also has to be reachable from where the workers are. By default it only listens on its own machine (`HOST=127.0.0.1`). Either set `HOST=0.0.0.0` and open its port, or better, put it behind your web server with HTTPS (the same way as JamTube), at an address like `https://lyrics.example.com`. Only the `/workers` addresses need to be reachable, and they only work with `WORKER_KEY`.

`GET /health` on the server lists the workers connected right now (`workers`).

## Running a worker

The worker needs the server's address and its `WORKER_KEY`, as `--server` and `--key`, or as the environment variables `JAMTUBE_LYRICS_URL` and `WORKER_KEY`. `--name` (or `WORKER_NAME`) is what the server calls it, which defaults to the computer's name. It shows in the server's log and in each song's Song Info ("Vocal separator: …, on gaming-pc").

The first run sets everything up in a `.venv` folder here, with the GPU build when there's an NVIDIA GPU. Later runs start straight away. It needs Python 3.10 to 3.13, and ffmpeg. It says what it's running on when it starts: an NVIDIA GPU (CUDA), a Mac's GPU (Core ML), or the CPU.

**macOS or Linux**

```sh
git clone https://github.com/WASDHarrison/jamtube-lyrics-worker && cd jamtube-lyrics-worker
./run.sh --server https://lyrics.example.com --key <WORKER_KEY>
```

On a Mac, get them with Homebrew: `brew install python@3.12 ffmpeg`. Apple Silicon Macs use their GPU (Metal) automatically.

**Windows** (PowerShell)

```powershell
git clone https://github.com/WASDHarrison/jamtube-lyrics-worker; cd jamtube-lyrics-worker
powershell -ExecutionPolicy Bypass -File run.ps1 --server https://lyrics.example.com --key <WORKER_KEY>
```

Get them with `winget install Python.Python.3.12 Gyan.FFmpeg`, then open a new window.

**Cloud machines**

On a GPU machine you rent by the hour, the worker is the only thing to start, and it's fine to switch the machine off at any time: its song goes to another worker or back to the server.

- **GCP, AWS, Azure** (a VM with an NVIDIA GPU): use a "deep learning" or GPU-ready image (it has the NVIDIA driver and Docker with GPU support), then build and run the container below. Or, on any Linux VM with the NVIDIA driver, clone this and use `./run.sh`.
- **Vast.ai, RunPod** and the like: their machines are containers already, usually with PyTorch and Python. Pick an image with CUDA, open its terminal, install ffmpeg (`apt-get install -y ffmpeg`), clone this and use `./run.sh`.

**Docker**

```sh
docker build -t jamtube-lyrics-worker .
docker run -d --restart unless-stopped --gpus all \
  -e JAMTUBE_LYRICS_URL=https://lyrics.example.com -e WORKER_KEY=<WORKER_KEY> -e WORKER_NAME=cloud-gpu \
  -v jamtube-models:/models jamtube-lyrics-worker
```

That's for an NVIDIA GPU, which needs the NVIDIA Container Toolkit; GPU images on the clouds usually have it. Without a GPU: `docker build --build-arg KIND=cpu -t jamtube-lyrics-worker:cpu .`, and leave out `--gpus all`. The volume keeps the model between runs. Without it, the model is downloaded each time the container starts.

Stop a worker with Ctrl+C (or `docker stop`): a song it was working on goes straight to another worker or back to the server.

## Settings

| Option | Environment | Default | |
|---|---|---|---|
| `--server` | `JAMTUBE_LYRICS_URL` | | The server's address. |
| `--key` | `WORKER_KEY` | | The server's `WORKER_KEY`. |
| `--name` | `WORKER_NAME` | the computer's name | What the server calls it. |
| `--models-dir` | `MODELS_DIR` | the user's cache folder | Where the model is kept between runs. |

How often a worker checks in, and how long one can go quiet before its song goes elsewhere, are set on the server (`WORKER_POLL_SECONDS`, `WORKER_HEARTBEAT_SECONDS`, `WORKER_TIMEOUT_SECONDS`; see its README).

## How it talks to the server

Plain HTTPS, always started by the worker. It works behind home routers, firewalls and cloud networks with nothing opened on the worker's side.

- `POST /workers/jobs/next` `{ name }`: asks for a song. The server holds the request until it has one (up to `WORKER_POLL_SECONDS`), and answers `204` when it has none, so the worker asks again. A worker waiting here is free.
- `GET /workers/jobs/:id/audio`: the song's parts to take the vocals out of (FLAC, lossless, 16-bit, 44.1 kHz).
- `POST /workers/jobs/:id/heartbeat`: every `heartbeatSeconds` while on it. `410` means the server doesn't want it anymore (the song was dropped, or went elsewhere), so the worker drops it.
- `POST /workers/jobs/:id/result`: the instrumental (FLAC).
- `POST /workers/jobs/:id/failed` `{ error }`: couldn't, so the song goes to another worker or back to the server.

Each needs `Authorization: Bearer <WORKER_KEY>`.

## Files

```
worker.py               The worker
run.sh                  Sets it up and runs it on macOS and Linux
run.ps1                 The same on Windows
requirements-cpu.txt    What it needs without an NVIDIA GPU (Macs too)
requirements-gpu.txt    What it needs with one (CUDA)
Dockerfile              The worker in a container, GPU or CPU
```
