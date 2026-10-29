# A worker in a container, for cloud machines (GCP, AWS, Azure, Vast.ai, RunPod...). Built on
# every push to main (.github/workflows/image.yml) as ghcr.io/wasdharrison/jamtube-lyrics-worker:
# `latest` with an NVIDIA GPU, `cpu` without one:
#
#   docker run --gpus all -e JAMTUBE_LYRICS_URL=https://lyrics.example.com -e WORKER_KEY=... ghcr.io/wasdharrison/jamtube-lyrics-worker
#
# Or build it: docker build -t jamtube-lyrics-worker . (without a GPU: --build-arg KIND=cpu)
# The model is downloaded into /models the first time; mount a volume there to keep it.
ARG KIND=gpu
ARG GPU_BASE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ARG CPU_BASE=ubuntu:22.04

FROM ${GPU_BASE} AS gpu
FROM ${CPU_BASE} AS cpu

FROM ${KIND}
ARG KIND
RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3 python3-pip ffmpeg \
  && rm -rf /var/lib/apt/lists/*
WORKDIR /worker
COPY requirements-${KIND}.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements-${KIND}.txt
COPY worker.py ./
ENV MODELS_DIR=/models PYTHONUNBUFFERED=1
ENTRYPOINT ["python3", "worker.py"]
