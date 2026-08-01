# syntax=docker/dockerfile:1.7

FROM node:20-bookworm-slim AS frontend-build
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

ARG VLLM_IMAGE=vllm/vllm-openai:v0.15.1
FROM ${VLLM_IMAGE} AS runtime
ENTRYPOINT []
WORKDIR /app

COPY pyproject.toml README.md ./
COPY driftsql/ ./driftsql/
RUN python -m pip install --no-cache-dir ".[service]"

COPY configs/ ./configs/
COPY scripts/serve_service.sh ./scripts/serve_service.sh
COPY --from=frontend-build /build/frontend/dist ./frontend/dist

ENV PYTHONUNBUFFERED=1 \
    DRIFTSQL_SERVICE_ENVIRONMENT=production \
    DRIFTSQL_SERVICE_HOST=0.0.0.0 \
    DRIFTSQL_SERVICE_PORT=8000 \
    DRIFTSQL_SERVICE_SERVE_FRONTEND=true
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "-m", "uvicorn", "driftsql.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
