import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// GH Pages serves the site under https://<user>.github.io/pde-agent-bench/.
// Builds use that base path so all asset URLs are correct; dev mode keeps "/".
const REPO_BASE = "/pde-agent-bench/";

export default defineConfig(({ command }) => ({
  base: command === "build" ? REPO_BASE : "/",
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    // Build output goes straight to docs/ at the repo root so GH Pages can
    // serve from the main branch's /docs folder.
    outDir: path.resolve(__dirname, "../../docs"),
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
}));
