# Deployment, authentication, and CI

## Docker GPU service

The production image installs the Python CLI/API package and runs the
FastAPI/vLLM service from a pinned vLLM base image. Models, adapters, reports
and databases are runtime mounts and are never copied into the image.
The image uses DriftSQL's portable tool-contract implementation; full VERL is
only required by training jobs, where the same tools automatically bind to the
pinned native VERL classes and rollout tracing.

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
export DRIFTSQL_AUTH_COOKIE_SECURE=true
docker compose up
```

The CLI sends the API key only through an authorization header. Legacy cookie
authentication endpoints remain available for API compatibility, but no Web
frontend is shipped. Clients may send either header form to `/api/*`:

```text
Authorization: Bearer <key>
X-DriftSQL-API-Key: <key>
```

`/auth/status`, `/auth/login`, `/auth/logout`, and `/health` remain public;
protected data is unavailable without a valid cookie or header.

The API key is a Pydantic `SecretStr`, is compared with a constant-time
function, and is never serialized into OpenAPI, health responses or logs.

## CI

`.github/workflows/ci.yml` runs three independent jobs:

- Python 3.11 installation, portable drift/unit tests, and source compilation;
- CLI/service installation, API integration tests and CLI entrypoint checks;
- a Docker BuildKit build of the Python/vLLM runtime and validation of the
  Compose deployment contract.

GPU inference, private model weights, BIRD databases and sealed Gates are not
downloaded in public CI. Their verification remains in the explicit local
acceptance and hash-audit scripts. CI builds the runtime image and validates
the Compose contract, but does not start a GPU-backed inference process.
