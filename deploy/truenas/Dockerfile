# syntax=docker/dockerfile:1.7

ARG LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-vulkan
ARG WHISPER_IMAGE=ghcr.io/ggml-org/whisper.cpp:main-vulkan

FROM ${LLAMA_IMAGE} AS llama_runtime
FROM ${WHISPER_IMAGE}

ARG SOURCE_COMMIT=unknown

LABEL org.opencontainers.image.title="Radio Archive Intelligence"
LABEL org.opencontainers.image.description="Evidence-first radio archive Web app for TrueNAS Apps"
LABEL org.opencontainers.image.source="https://github.com/Naelus/broadcastify-cli"
LABEL org.opencontainers.image.revision="${SOURCE_COMMIT}"

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=llama_runtime /app /opt/llama

WORKDIR /opt/radio-archive
COPY pyproject.toml README.md LICENSE ./
COPY broadcastify_cli ./broadcastify_cli

RUN python3 -m venv /opt/radio-archive-venv \
    && /opt/radio-archive-venv/bin/python -m pip install --no-cache-dir --upgrade pip \
    && /opt/radio-archive-venv/bin/python -m pip install --no-cache-dir \
        ".[analysis,portable-diarization]"

ENV PATH="/opt/radio-archive-venv/bin:/app/build/bin:/opt/llama:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FFMPEG_PATH=/usr/bin/ffmpeg \
    WHISPER_CPP_PATH=/app/build/bin/whisper-cli \
    LLAMA_SERVER_PATH=/opt/llama/llama-server

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["/opt/radio-archive-venv/bin/python", "-c", "import json, urllib.request; value=json.load(urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=3)); assert value == {'status': 'ok', 'scope': 'trusted-lan'}"]

ENTRYPOINT ["/opt/radio-archive-venv/bin/python", "-m", "broadcastify_cli.web_app"]
CMD ["--host", "0.0.0.0", "--port", "8765", "--working-dir", "/data", "--output-dir", "/data/archives", "--database", "/data/archives/broadcastify-analysis.sqlite3"]
