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
export DRIFTSQL_AUTH_COOKIE_SECURE=true
docker compose up
```

The Chinese Studio exchanges the API key once at `/auth/login` for a
short-lived, random, HttpOnly, SameSite=Strict cookie. The key is not written
to localStorage/sessionStorage and the same cookie authenticates JSON and SSE
traffic. `/auth/logout` revokes the server-side token. The default lifetime is
eight hours and can be changed with
`DRIFTSQL_SERVICE_AUTH_SESSION_TTL_SECONDS`.

Non-browser clients may continue to send either header form to `/api/*`:

```text
Authorization: Bearer <key>
X-DriftSQL-API-Key: <key>
```

`/auth/status`, `/auth/login`, `/auth/logout`, `/health`, and static Studio
assets remain public; protected data is still unavailable without a valid
cookie or header. Browser sessions are kept in memory, so production must use
one API process (the supplied deployment) or add a shared session store before
enabling multiple FastAPI workers.

The API key is a Pydantic `SecretStr`, is compared with a constant-time
function, and is never serialized into OpenAPI, health responses or logs.

## CI

`.github/workflows/ci.yml` runs three independent jobs:

- Python 3.11 installation, portable drift/unit tests, and source compilation;
- Node 20 clean installation from `package-lock.json`, frontend tests, and a
  production TypeScript/Vite build.
- a Docker BuildKit build of the reproducible frontend stage and validation of
  the Compose deployment contract.

GPU inference, private model weights, BIRD databases and sealed Gates are not
downloaded in public CI. Their verification remains in the explicit local
acceptance and hash-audit scripts.

The full GPU runtime image cannot be built on the current workstation because
its Snap Docker installation is unhealthy. CI therefore validates the
lightweight image stage and Compose syntax; a successful CI run is still
required after the repository is pushed.
