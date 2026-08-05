import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// `base` is empty by default for local dev (BASE_URL = '/'). For the public
// droplet deploy, build with VITE_BASE=/vps/app/ so assets resolve correctly
// behind the nginx subroute. The api clients independently respect VITE_API_BASE.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: process.env.VITE_BASE || '/',
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
    proxy: {
      '/api': process.env.VITE_API_TARGET || 'http://localhost:8000',
      '/ws': {
        target: (process.env.VITE_API_TARGET || 'http://localhost:8000')
          .replace(/^http/, 'ws'),
        ws: true,
      },
    },
  },
})
