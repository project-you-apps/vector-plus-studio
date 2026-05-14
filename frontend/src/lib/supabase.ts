import { createBrowserClient } from '@supabase/ssr'

// Cookie domain `.project-you.app` is load-bearing for SSO across Waving Cat
// apps (VPS, Membot, Membraine, future marketplace) hosted under subroutes of
// project-you.app. Leading dot is required so the session cookie is readable
// from any subroute/subdomain. In dev (localhost) we leave it undefined so the
// browser falls back to host-only cookies.
const isProd = import.meta.env.PROD

const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY

if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
  throw new Error(
    'Missing VITE_SUPABASE_URL or VITE_SUPABASE_ANON_KEY. ' +
    'Set both in .env.local (dev) or .env.production (build).',
  )
}

export const supabase = createBrowserClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
  cookieOptions: {
    ...(isProd ? { domain: '.project-you.app' } : {}),
    path: '/',
    sameSite: 'lax',
    secure: isProd,
  },
})
