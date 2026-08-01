# DriftSQL Studio

DriftSQL P3 的中文 React/TypeScript 智能体运行与审计工作台。

```bash
npm ci
npm run dev       # Vite development server, proxies FastAPI on :8000
npm run typecheck
npm test
npm run build     # output: frontend/dist, served by FastAPI
```

The browser receives only public scenario data and persisted trajectory
events. Gold SQL and expected result fingerprints remain server-side.

For the integrated production surface, run this from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0,2 bash scripts/serve_studio.sh
```

Then open `http://127.0.0.1:8000`. Set
`DRIFTSQL_SKIP_FRONTEND_BUILD=1` to reuse an already verified `dist` bundle.
