import { useQuery } from "@tanstack/react-query";
import { fallbackData } from "./fallback";

export type Stats = {
  totalCases: number;
  pdeTypes: number;
  backends: number;
  modelsEvaluated: number;
  agents: number;
  version: string;
};

export type Backend = "dolfinx" | "firedrake" | "dealii";

export type LeaderboardRow = {
  id: number;
  model: string;
  provider: string;
  agent: string | null;
  backend: Backend;
  passRate: number;
  accuracyGate: number | null;
  timeGate: number | null;
  l2Error: number | null;
  costPer1K: number | null;
  date: string | null;
  pdeScores: Record<string, number>;
};

export type PdeType = {
  id: string;
  name: string;
  abbr: string;
  category: string;
  color: string;
  bgColor: string;
  cases: number;
  equation: string;
  description: string;
  backends: string[];
  backendCases: Record<string, number | null>;
};

export type ModelMeta = {
  id: string;
  displayName: string;
  provider: string;
  family: "LLM" | "Agent";
  paramSize: string | null;
  notes: string | null;
};

export type Author = {
  name: string;
  affiliation: string;
  homepage: string | null;
  isCorresponding: boolean;
  isEqual?: boolean;
};

export type AuthorsResponse = {
  authors: Author[];
  affiliations: string[];
};

export type Finding = {
  n: number;
  title: string;
  body: string;
  icon: string;
};

export type Figure = {
  id: string;
  src: string;
  thumb: string;
  caption: string;
  section: string;
};

export type Citation = {
  title: string;
  venue: string;
  year: number;
  abstract: string;
  bibtex: string;
};

// Resolved at build/dev time. In dev, Vite proxies /api → FastAPI (see vite.config.ts).
// In prod, set VITE_API_BASE to a deployed FastAPI origin to enable live data;
// leave it unset to ship a static-only build that always serves bundled fallback.
const BASE = import.meta.env.VITE_API_BASE ?? "/api";
const HAS_REMOTE_API = Boolean(import.meta.env.VITE_API_BASE) || import.meta.env.DEV;

async function get<T>(path: string, fallback: T): Promise<T> {
  if (!HAS_REMOTE_API) return fallback;
  try {
    const res = await fetch(`${BASE}${path}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return (await res.json()) as T;
  } catch (err) {
    console.warn(`[api] ${path} failed, using fallback`, err);
    return fallback;
  }
}

export const useStats = () =>
  useQuery({ queryKey: ["stats"], queryFn: () => get<Stats>("/stats", fallbackData.stats) });

export const useLeaderboard = (backend: string = "all") =>
  useQuery({
    queryKey: ["leaderboard", backend],
    queryFn: () =>
      get<LeaderboardRow[]>(
        `/leaderboard${backend === "all" ? "" : `?backend=${backend}`}`,
        fallbackData.leaderboard.filter(
          (r) => backend === "all" || r.backend === backend,
        ) as LeaderboardRow[],
      ),
  });

export const usePdeTypes = () =>
  useQuery({
    queryKey: ["pde-types"],
    queryFn: () => get<PdeType[]>("/pde-types", fallbackData.pdeTypes as PdeType[]),
  });

export const useModels = () =>
  useQuery({
    queryKey: ["models"],
    queryFn: () => get<ModelMeta[]>("/models", fallbackData.models as ModelMeta[]),
  });

export const useAuthors = () =>
  useQuery({
    queryKey: ["authors"],
    queryFn: () => get<AuthorsResponse>("/authors", fallbackData.authors as AuthorsResponse),
  });

export const useFindings = () =>
  useQuery({
    queryKey: ["findings"],
    queryFn: () => get<Finding[]>("/findings", fallbackData.findings as Finding[]),
  });

export const useFigures = () =>
  useQuery({
    queryKey: ["figures"],
    queryFn: () => get<Figure[]>("/figures", fallbackData.figures as Figure[]),
  });

export const useCitation = () =>
  useQuery({
    queryKey: ["citation"],
    queryFn: () => get<Citation>("/citation", fallbackData.citation as Citation),
  });

/** Resolve an asset path (e.g. "figures/intro.png") against the Vite base URL,
 *  so the same JSON works in dev (base "/") and on GH Pages (base "/<repo>/"). */
export function withBase(path: string): string {
  const trimmed = path.replace(/^\//, "");
  return `${import.meta.env.BASE_URL}${trimmed}`;
}
