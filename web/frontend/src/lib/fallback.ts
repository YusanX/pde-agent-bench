// Static mirror of /api/* responses so the site still renders if the
// FastAPI backend is unreachable. Keep in sync with web/server/app/data/mock.json.
import data from "../../../server/app/data/mock.json";

export const fallbackData = {
  stats: data.stats,
  leaderboard: data.leaderboard,
  pdeTypes: data.pdeTypes,
  models: data.models,
  authors: data.authors,
  findings: data.findings,
  figures: data.figures,
  citation: data.citation,
};
