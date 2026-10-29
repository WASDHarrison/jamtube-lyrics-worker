# A worker in a container, for cloud machines (GCP, AWS, Azure, Vast.ai, RunPod...). With an
# NVIDIA GPU (the default):
#
#   docker build -t jamtube-lyrics-worker .
#   docker run --gpus all -e JAMTUBE_LYRICS_URL=https://lyrics.example.com -e WORKER_KEY=... jamtube-lyrics-worker
#
# Without one: docker build --build-arg KIND=cpu -t jamtube-lyrics-worker:cpu .
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
