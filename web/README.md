# PDEAgent-Bench Showcase Site

A polished academic project page for **PDEAgent-Bench**, modelled after
<https://realpdebench.github.io/>. Two pieces:

| Folder        | Stack                                                       | Port |
| ------------- | ----------------------------------------------------------- | ---- |
| [frontend/](frontend/) | Vite + React + TypeScript + Tailwind + shadcn-style UI + Recharts + Framer Motion + TanStack Query | 5173 |
| [server/](server/)   | FastAPI (mock data in `app/data/mock.json`)                  | 8000 |

The frontend dev server proxies `/api/*` to the FastAPI backend, so you can
treat the whole thing as a single origin in the browser. If the backend isn't
running, the frontend falls back to a static mirror of `mock.json` bundled
with the build — every section still renders.

---

## Quick start

Two terminals:

```bash
# Terminal 1 — backend
cd web/server
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# Swagger docs at http://localhost:8000/docs
```

```bash
# Terminal 2 — frontend
cd web/frontend
npm install
npm run dev
# Site at http://localhost:5173
```

## Production build (GitHub Pages)

The Vite build is wired to write straight into the repo's `docs/` folder so
GH Pages can serve it from the main branch.

```bash
cd web/frontend
npm run build      # → ../../docs/   (replaces previous build)
```

Then in the repo's GitHub settings → **Pages** → set source to `Deploy from
a branch`, branch `main`, folder `/docs`. The site will be live at
`https://<owner>.github.io/pde-agent-bench/`.

In production builds the API client (`src/lib/api.ts`) detects
`import.meta.env.DEV === false` and goes straight to the bundled
`mock.json` fallback — no FastAPI server is contacted. Edit
[server/app/data/mock.json](server/app/data/mock.json) and re-run
`npm run build` to update the deployed numbers.

The build's GH Pages base path is hard-coded in [vite.config.ts](frontend/vite.config.ts)
as `/pde-agent-bench/`. If the repo gets renamed, update that constant.

## Where things live

```
web/
├── frontend/
│   ├── public/figures/           ← rasterized paper figures (intro, method, …)
│   ├── src/
│   │   ├── App.tsx               ← page composition
│   │   ├── lib/
│   │   │   ├── api.ts            ← typed fetch + react-query hooks
│   │   │   └── fallback.ts       ← imports server/app/data/mock.json
│   │   ├── components/           ← reusable atoms (AnimatedCounter, SectionHeader)
│   │   └── sections/             ← Hero, Authors, Abstract, Leaderboard, …
│   ├── tailwind.config.ts        ← design tokens (brand-*, accent-*, ink-*)
│   ├── vite.config.ts            ← /api → :8000 proxy
│   └── package.json
└── server/
    ├── app/
    │   ├── main.py               ← FastAPI app + CORS + router wiring
    │   ├── schemas.py            ← Pydantic response models
    │   ├── store.py              ← loads mock.json once
    │   ├── data/mock.json        ← single source of truth for everything
    │   └── routers/              ← stats, leaderboard, models, pde-types, findings, …
    └── requirements.txt
```

## API

| Method | Path                          | Description                             |
| ------ | ----------------------------- | --------------------------------------- |
| GET    | `/api/health`                 | liveness check                          |
| GET    | `/api/stats`                  | hero counters                           |
| GET    | `/api/leaderboard?backend=…`  | sortable leaderboard rows               |
| GET    | `/api/leaderboard/{id}`       | single row + per-PDE breakdown          |
| GET    | `/api/models`                 | model + agent catalog                   |
| GET    | `/api/pde-types`              | 12 PDE families                         |
| GET    | `/api/findings`               | numbered insight cards                  |
| GET    | `/api/authors`                | research team + affiliations            |
| GET    | `/api/figures`                | paper-figure gallery metadata           |
| GET    | `/api/citation`               | title, venue, abstract, BibTeX          |

All shapes mirrored on the frontend in `frontend/src/lib/api.ts`. Edit
`server/app/data/mock.json` to update content; restart uvicorn to pick up
changes (or just re-run `npm run dev` — the fallback also reads the same
file).

## Wiring real benchmark numbers

When you're ready to plug in real data, replace `server/app/store.py` with a
loader that reads from `data/benchmark_v2.jsonl` plus your run summaries.
Keep the response shapes (`schemas.py`) intact and the frontend won't need
to change.

## Replacing the old `docs/` prototype

The legacy single-page site in [docs/](../docs/) is left untouched. After this
new site is approved, you can either:

1. Configure GitHub Pages to publish `web/frontend/dist/` instead of `docs/`, or
2. Run `npm run build` and copy `dist/*` over the contents of `docs/`.

## Scripts

`scripts/rasterize_figures.py` — converts the chosen NIPS26_pdeagentbench PDF
figures (error margin, error/time scatter, radar chart) into PNGs under
`web/frontend/public/figures/`. Re-run it any time the paper figures change.
