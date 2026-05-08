export function cn(...args: Array<string | false | null | undefined>): string {
  return args.filter(Boolean).join(" ");
}

export function fmtPct(n: number, digits = 1): string {
  return `${n.toFixed(digits)}%`;
}

export function fmtNum(n: number, digits = 4): string {
  return n.toFixed(digits);
}

export function fmtMoney(n: number): string {
  return `$${n.toFixed(2)}`;
}

export function rankColor(passRate: number): string {
  if (passRate >= 68) return "#10b981";
  if (passRate >= 58) return "#f59e0b";
  return "#ef4444";
}

export const BACKEND_LABELS: Record<string, string> = {
  all: "All Backends",
  dolfinx: "DOLFINx",
  firedrake: "Firedrake",
  dealii: "deal.II",
};

export const BACKEND_COLORS: Record<string, string> = {
  dolfinx: "#3f60e6",
  firedrake: "#22d3ee",
  dealii: "#a78bfa",
};
