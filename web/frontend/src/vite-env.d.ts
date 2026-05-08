/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Origin of a deployed FastAPI server (e.g. https://api.example.com).
   *  Unset → static-only build serving bundled fallback data.
   *  Dev → ignored (Vite proxies /api to localhost:8000). */
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
