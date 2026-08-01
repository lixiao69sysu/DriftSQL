# Deployment, authentication, and CI

## Docker GPU service

The production image builds the React Studio in a Node 20 stage and runs the
FastAPI/vLLM service from a pinned vLLM base image. Models, adapters, reports
and databases are runtime mounts and are never copied into the image.

```bash
export CUDA_VISIBLE_DEVICES=0,2
export DRIFTSQL_TP=2
docker compose up --build
```

The default host port is 8001. The container drops Linux capabilities, enables
`no-new-privileges`, exposes only the application port, and keeps the model,
checkpoint and report mounts read-only. The data mount remains writable for
the per-Session SQLite sandbox and durable trajectory repository.

## API authentication

Authentication is optional for local development and should be enabled behind
TLS in production:

```bash
export DRIFTSQL_AUTH_ENABLED=true
export DRIFTSQL_SERVICE_API_KEY='generate-a-long-random-secret'
docker compose up
```

All `/api/*` requests then require either header form:

```text
Authorization: Bearer <key>
X-DriftSQL-API-Key: <key>
```

`/health` and static Studio assets remain public for container health checks.
The API key is a Pydantic `SecretStr`, is compared with a constant-time
function, and is never serialized into OpenAPI, health responses or logs.

## CI

`.github/workflows/ci.yml` runs two independent jobs:

- Python 3.11 installation, portable drift/unit tests, and source compilation;
- Node 20 clean installation from `package-lock.json`, frontend tests, and a
  production TypeScript/Vite build.

GPU inference, private model weights, BIRD databases and sealed Gates are not
downloaded in public CI. Their verification remains in the explicit local
acceptance and hash-audit scripts.
