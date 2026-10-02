# syntax=docker/dockerfile:1.7

ARG LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-vulkan@sha256:ffa3e6a781e3cffbd5975be4b9062854689b3603165366e33358f88270da2c73
ARG WHISPER_IMAGE=ghcr.io/ggml-org/whisper.cpp:main-vulkan@sha256:d991b7cec9528a483202c50c4c367e58a6b7c3a69946d9d963355a83adb0a36a

FROM ${WHISPER_IMAGE} AS whisper_runtime
FROM ${LLAMA_IMAGE}

ARG SOURCE_COMMIT

LABEL org.opencontainers.image.title="Radio Archive Intelligence"
LABEL org.opencontainers.image.description="Evidence-first radio archive Web app for TrueNAS Apps"
LABEL org.opencontainers.image.source="https://github.com/Naelus/broadcastify-cli"
LABEL org.opencontainers.image.revision="${SOURCE_COMMIT}"

USER root

# A deployable image must identify the exact committed source that produced it.
# Refuse an omitted, shortened, or non-Git revision before doing the expensive
# package installation so About/runtime metadata can never silently say
# "unknown" in a managed deployment.
RUN set -eu; \
    commit="${SOURCE_COMMIT}"; \
    test "${#commit}" -eq 40; \
    case "${commit}" in *[!0-9a-f]*) exit 1 ;; esac

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        libgomp1 \
        python3-pip \
        python3-venv \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

# The official Whisper image installs under /usr/local. Its ggml libraries
# have the same names as llama.cpp's but a different ABI. Scope the loader
# path to this executable and keep the historical configured CLI path valid.
COPY --from=whisper_runtime /usr/local/bin/whisper-cli /opt/whisper/bin/whisper-cli
COPY --from=whisper_runtime /usr/local/lib/ /opt/whisper/lib/
RUN mkdir -p /app/build/bin \
    && printf '%s\n' '#!/bin/sh' \
       'export LD_LIBRARY_PATH="/opt/whisper/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"' \
       'exec /opt/whisper/bin/whisper-cli "$@"' > /app/build/bin/whisper-cli \
    && chmod 755 /app/build/bin/whisper-cli

WORKDIR /opt/radio-archive
COPY pyproject.toml README.md LICENSE ./
COPY broadcastify_cli ./broadcastify_cli

# Build hosts may use a restrictive umask for private deployment workspaces.
# Normalize only the public application tree so the unprivileged runtime user
# can import it without weakening persistent /data permissions.
RUN chmod -R a+rX /opt/radio-archive

RUN python3 -m venv /opt/radio-archive-venv \
    && /opt/radio-archive-venv/bin/python -m pip install --no-cache-dir --upgrade pip \
    && /opt/radio-archive-venv/bin/python -m pip install --no-cache-dir \
        ".[analysis,portable-diarization]"

ENV PATH="/opt/radio-archive-venv/bin:/app/build/bin:/app:${PATH}" \
    LD_LIBRARY_PATH=/app \
    BROADCASTIFY_SOURCE_COMMIT="${SOURCE_COMMIT}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FFMPEG_PATH=/usr/bin/ffmpeg \
    WHISPER_CPP_PATH=/app/build/bin/whisper-cli \
    LLAMA_SERVER_PATH=/app/llama-server

EXPOSE 8765
EXPOSE 48765/udp

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["/opt/radio-archive-venv/bin/python", "-c", "import json, urllib.request; value=json.load(urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=3)); assert value == {'status': 'ok', 'scope': 'trusted-lan'}"]

ENTRYPOINT ["/opt/radio-archive-venv/bin/python", "-m", "broadcastify_cli.web_app"]
CMD ["--host", "0.0.0.0", "--port", "8765", "--working-dir", "/data", "--output-dir", "/data/archives", "--database", "/data/archives/broadcastify-analysis.sqlite3"]
