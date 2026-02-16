/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL: string;
  readonly VITE_ENV: string;
  readonly VITE_ENABLE_DEMO_MODE: string;
  readonly VITE_ENABLE_REALTIME_SIMULATION: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
