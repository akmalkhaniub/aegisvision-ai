# Multi-Stage Production Dockerfile for AegisVision AI (CPU & Cloud Run NVIDIA L4 GPU Compatible)
FROM python:3.12-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim AS runtime
WORKDIR /app

# Create non-root operator user
RUN groupadd -r aegis && useradd -r -g aegis aegis

COPY --from=builder /install /usr/local
COPY src ./src
COPY pyproject.toml README.md ./

USER aegis
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/v1/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
