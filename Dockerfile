FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FLUXOFORUM_DATA_ROOT=/workspace/fluxoforum-data \
    HF_HOME=/workspace/fluxoforum-data/models \
    HUGGINGFACE_HUB_CACHE=/workspace/fluxoforum-data/models/hub

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git python3 python3-pip python3-venv curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/fluxoforum
COPY . .
RUN python3 -m pip install --break-system-packages --upgrade pip setuptools wheel \
    && python3 -m pip install --break-system-packages -r requirements.txt \
    && python3 -m pip install --break-system-packages --no-deps -e .

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 scripts/healthcheck.py

CMD ["bash", "scripts/launch.sh"]

