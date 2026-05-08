# PDEAgent-Bench Mock API

FastAPI app that serves mock benchmark data to the showcase frontend.
All data lives in [app/data/mock.json](app/data/mock.json) — replace it with
real benchmark run results when ready.

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Then open <http://localhost:8000/docs> for interactive Swagger UI.

## Endpoints

| Method | Path                        | Description                                         |
| ------ | --------------------------- | --------------------------------------------------- |
| GET    | `/api/health`               | Liveness check                                      |
| GET    | `/api/stats`                | Hero counters (totalCases, pdeTypes, ...)           |
| GET    | `/api/leaderboard`          | Sortable leaderboard rows (`?backend=`, `?sort=`)   |
| GET    | `/api/leaderboard/{id}`     | Single row                                          |
| GET    | `/api/models`               | Model + agent catalog                               |
| GET    | `/api/pde-types`            | 12 PDE families with equations                      |
| GET    | `/api/findings`             | Numbered insight cards                              |
| GET    | `/api/authors`              | Research team + affiliations                        |
| GET    | `/api/figures`              | Paper figure gallery metadata                       |
| GET    | `/api/citation`             | Title, venue, abstract, BibTeX                      |

## Replacing mock data

Swap out `app/data/mock.json` for real numbers, or change `app/store.py` to
load from the actual benchmark output (e.g. `data/benchmark_v2.jsonl` plus run
summaries). Schemas in `app/schemas.py` are the contract — keep them aligned
with the frontend's `src/lib/api.ts`.
