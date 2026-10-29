# JamTube Lyrics Worker

Takes the vocals out of songs for a [JamTube Lyrics](https://github.com/WASDHarrison/jamtube-lyrics) server, on another computer, ideally one with a GPU. Removing the vocals is the slow part of making a karaoke song: a couple of minutes on a small server's CPU, seconds on a GPU.

Run as many workers as you like, wherever you like: your laptop, a gaming PC, a cloud GPU for the night. Each one connects to the server, asks for a song, takes its vocals out, sends the result back and asks for the next. The server gives each song to a free worker, soonest-needed first, and does it itself when none is free. If a worker fails, is stopped or goes quiet (switched off, lost its connection), its song goes to another worker, or back to the server. So a worker can come and go at any time, and nothing breaks.

A worker keeps nothing: each song is downloaded into the worker's own temporary folder and deleted once it's sent back, and the folder goes when the worker stops. (A worker killed outright, without the chance to stop, leaves its folder to the system, which clears temporary files; a cloud machine or container takes it with it.) Only the model (about 60 MB) is kept, so it's downloaded once. A worker gets it ready as soon as it starts (it asks the server which model it uses), and says "Ready for songs.", so its first song doesn't wait for it. It removes the vocals exactly as the server would, with the same separator, model and settings, which the server sends with each song. The result is the same, sample for sample.

## On the server

Set `WORKER_KEY` in JamTube Lyrics' `.env` to a long random secret (`openssl rand -hex 32`), and restart it. The server also has to be reachable from where the workers are. By default it only listens on its own machine (`HOST=127.0.0.1`). Either put it behind your web server with HTTPS (the same way as JamTube), at an address like `https://lyrics.example.com`, or set `HOST=0.0.0.0` and open its port. Only the `/workers` addresses need to be reachable, and they only work with `WORKER_KEY`. Behind a web server, have it pass on `X-Forwarded-Proto` (nginx: `proxy_set_header X-Forwarded-Proto $scheme;`), so workers started from the server are given its `https://` address.

Workers start from the server, which hands out this worker: keep a copy of this repository next to JamTube Lyrics' folder (`git clone` it beside it, and `git pull` it when you update the server), or say where it is with `WORKER_FILES_DIR`.

`GET /health` on the server lists the workers connected right now (`workers`).

## Starting a worker

One line, on any computer, with nothing to install first. It gets the worker from the server, sets everything up the first time (a few minutes: Python, and what vocal removal needs, a few GB, with the GPU build when there's an NVIDIA GPU) and starts it. After that, the same line starts it in seconds, with the server's latest worker.

**macOS or Linux**

```sh
curl -fsSL https://lyrics.example.com/workers/start/<WORKER_KEY> | sh
```

**Windows** (PowerShell)

```powershell
irm https://lyrics.example.com/workers/start/<WORKER_KEY> | iex
```

Stop it with Ctrl+C: a song it was working on goes straight to another worker or back to the server. Options go after it: `... | sh -s -- --name gaming-pc` on macOS and Linux.

It says what it's running on when it starts: an NVIDIA GPU (CUDA), a Mac's GPU (Core ML), or the CPU. Everything goes in one folder, `~/.jamtube-lyrics-worker` (Windows: `%LOCALAPPDATA%\jamtube-lyrics-worker`; or `JAMTUBE_WORKER_DIR`): [uv](https://docs.astral.sh/uv/) and the Python it brings, what the worker needs, ffmpeg if the computer has none, and the model. Deleting the folder removes the worker completely.

**From a copy of this repository** (no server to get it from): `./run.sh --server https://lyrics.example.com --key <WORKER_KEY>`, or on Windows `powershell -ExecutionPolicy Bypass -File run.ps1 --server ... --key ...`. It's the same script, keeping everything in this folder.

**Cloud machines**

The worker is ready to run as a container: `ghcr.io/wasdharrison/jamtube-lyrics-worker`, `latest` for an NVIDIA GPU and `cpu` without one, built from `main` on every push. It's private like this repository, so a machine first logs in with a GitHub token that can read packages (a classic personal access token with `read:packages`): `docker login ghcr.io -u WASDHarrison` and the token as the password.

- **Vast.ai, RunPod** and the like: make a template with the image `ghcr.io/wasdharrison/jamtube-lyrics-worker`, the environment variables `JAMTUBE_LYRICS_URL` and `WORKER_KEY`, and the registry login (`ghcr.io`, your GitHub name and the token). A machine rented with it starts working as soon as it's up, and it's fine to destroy it any time: its song goes to another worker or back to the server. Or rent any machine with a terminal and use the one line above.
- **GCP, AWS, Azure** (a VM with an NVIDIA GPU): use a GPU-ready image (it has the NVIDIA driver and Docker with GPU support), log in as above, then:

```sh
docker run -d --restart unless-stopped --gpus all \
  -e JAMTUBE_LYRICS_URL=https://lyrics.example.com -e WORKER_KEY=<WORKER_KEY> -e WORKER_NAME=cloud-gpu \
  -v jamtube-models:/models ghcr.io/wasdharrison/jamtube-lyrics-worker
```

`--gpus all` needs the NVIDIA Container Toolkit, which GPU images on the clouds have. Without a GPU, use the `cpu` image and leave out `--gpus all`. The volume keeps the model between runs; without it, the model is downloaded each time the container starts. Stop it with `docker stop`. To build the image yourself: `docker build -t jamtube-lyrics-worker .` (without a GPU: `--build-arg KIND=cpu`).

## Settings

| Option | Environment | Default | |
|---|---|---|---|
| `--server` | `JAMTUBE_LYRICS_URL` | | The server's address. |
| `--key` | `WORKER_KEY` | | The server's `WORKER_KEY`. |
| `--name` | `WORKER_NAME` | the computer's name | What the server calls it. |
| `--models-dir` | `MODELS_DIR` | `models`, in its folder | Where the model is kept between runs. |

And for the start script (`run.sh`, `run.ps1`):

| Environment | Default | |
|---|---|---|
| `JAMTUBE_WORKER_DIR` | `~/.jamtube-lyrics-worker` (Windows: `%LOCALAPPDATA%\jamtube-lyrics-worker`) | Its folder, started from the server. |
| `PYTHON_VERSION` | `3.12` | The Python it runs on (3.10 to 3.13). |
| `UV_INSTALLER` | uv's installer from astral.sh | Where it gets uv the first time. |

How often a worker checks in, and how long one can go quiet before its song goes elsewhere, are set on the server (`WORKER_POLL_SECONDS`, `WORKER_HEARTBEAT_SECONDS`, `WORKER_TIMEOUT_SECONDS`; see its README).

## How it talks to the server

Plain HTTPS, always started by the worker. It works behind home routers, firewalls and cloud networks with nothing opened on the worker's side.

- `GET /workers/separator`: how the server removes vocals (`{ separator, model, overlap }`), so the worker can get the model ready when it starts.
- `POST /workers/jobs/next` `{ name }`: asks for a song. The server holds the request until it has one (up to `WORKER_POLL_SECONDS`), and answers `204` when it has none, so the worker asks again. A worker waiting here is free.
- `GET /workers/jobs/:id/audio`: the song's parts to take the vocals out of (FLAC, lossless, 16-bit, 44.1 kHz).
- `POST /workers/jobs/:id/heartbeat`: every `heartbeatSeconds` while on it. `410` means the server doesn't want it anymore (the song was dropped, or went elsewhere), so the worker drops it.
- `POST /workers/jobs/:id/result`: the instrumental (FLAC).
- `POST /workers/jobs/:id/failed` `{ error }`: couldn't, so the song goes to another worker or back to the server.

Each needs `Authorization: Bearer <WORKER_KEY>`.

## Files

```
worker.py               The worker
run.sh                  Sets it up and starts it on macOS and Linux (what the server hands out)
run.ps1                 The same on Windows
requirements-cpu.txt    What it needs without an NVIDIA GPU (Macs too)
requirements-gpu.txt    What it needs with one (CUDA)
Dockerfile              The worker in a container, GPU or CPU
.github/workflows/      Builds the container image on every push to main
```
