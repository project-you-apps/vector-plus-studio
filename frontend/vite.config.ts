import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { execSync } from 'node:child_process'

// ─── Build identity, injected at build/serve time ───────────────────────────
// WHY THIS EXISTS (2026-11-04). On 11-04 a provenance feature appeared not to work. The
// screen that "proved" it carried no indication of WHICH BUILD produced it or WHERE it
// was served from, and the feature's commit was unpushed -- so "the code is absent from
// this deploy" and "the feature is structurally impossible here" rendered identically.
// A screenshot with no build identity cannot be evidence about code.
//
// ⚠ The same gap was named on 2026-09-15 for the Restaurateur app ("no way to see which
// build a device is running without querying the database afterward") and deferred. It
// has now cost a diagnosis. Hence: injected, not optional.
function sh(cmd: string): string {
  try {
    // stderr ignored on purpose: this repo has unreadable dirs (.pytest_cache) that make
    // git chatter on stderr while still exiting 0.
    return execSync(cmd, { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim()
  } catch {
    return ''
  }
}

// ⭐ `dirty` is the load-bearing one, not `sha`. A dev build off uncommitted source whose
// footer shows a clean sha is a LIE of exactly the kind this stamp exists to prevent --
// it would read as "you are looking at commit X" when you are looking at X plus edits.
const BUILD_SHA = sh('git rev-parse --short HEAD') || 'unknown'
const BUILD_DIRTY = sh('git status --porcelain') !== ''

export default defineConfig(({ command }) => ({
  plugins: [react(), tailwindcss()],
  // `base` is empty by default for local dev (BASE_URL = '/'). For the public
  // droplet deploy, build with VITE_BASE=/vps/app/ so assets resolve correctly
  // behind the nginx subroute. The api clients independently respect VITE_API_BASE.
  base: process.env.VITE_BASE || '/',
  define: {
    __BUILD_SHA__: JSON.stringify(BUILD_SHA),
    __BUILD_DIRTY__: JSON.stringify(BUILD_DIRTY),
    __BUILD_TIME__: JSON.stringify(new Date().toISOString()),
    // 'serve' = vite dev server (the /api proxy below is live).
    // 'build' = a static bundle; there is NO proxy, and VITE_API_BASE governs instead.
    __BUILD_MODE__: JSON.stringify(command),
    // ⛔ ONLY MEANINGFUL WHEN __BUILD_MODE__ === 'serve'. In a static build the dev proxy
    // does not exist, so emitting a target here would describe a route that is not there.
    // posture.ts refuses to display it outside serve mode for that reason.
    __DEV_PROXY_TARGET__: JSON.stringify(
      command === 'serve' ? (process.env.VITE_API_TARGET || 'http://localhost:8000') : ''),
  },
  server: {
    // Backend target is env-driven so several full stacks can run side by side, each
    // frontend talking to its OWN backend:
    //
    //   VITE_API_TARGET=http://localhost:8001 npm run dev -- --port 5174
    //
    // Needed because `engine` is a process-level singleton with ONE mounted cart. Three
    // browsers against one backend give three real identities sharing one open cart, so
    // whoever mounts last silently changes what everyone else is looking at. Separate
    // processes are the only way to have three people on three different carts today.
    //
    // ⛔⛔ AND THIS PROXY IS WHY A FRONTEND-ONLY "LOCAL MODE" BADGE WOULD BE A LIE.
    // The browser only ever sees a RELATIVE `/api`. It cannot observe that the request
    // is forwarded to localhost:8000 -- nor that someone started the dev server with
    // VITE_API_TARGET=https://project-you.app, in which case the page is served from
    // the user's machine and every byte still goes to a public server. The browser
    // cannot tell those apart. ⇒ Only the BACKEND can say where the backend is, which
    // is why posture.ts trusts `status.public_host` over anything it computes itself.
    proxy: {
      '/api': process.env.VITE_API_TARGET || 'http://localhost:8000',
      '/ws': {
        target: (process.env.VITE_API_TARGET || 'http://localhost:8000')
          .replace(/^http/, 'ws'),
        ws: true,
      },
    },
  },
}))
