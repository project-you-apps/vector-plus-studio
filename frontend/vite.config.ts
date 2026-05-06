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
    proxy: {
      '/api': 'http://localhost:8000',
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
})
