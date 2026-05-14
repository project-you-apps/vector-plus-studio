import { create } from 'zustand'
import type { Session, User } from '@supabase/supabase-js'
import { supabase } from '../lib/supabase'

export type OAuthProvider = 'google' | 'github'

// This app's identifier for the central auth router. Each Waving Cat app sets
// its own constant ('vps', 'membot', etc). The router at project-you.app/
// reads this from localStorage post-OAuth and redirects to the correct app.
const APP_ID = 'vps'
const RETURN_APP_KEY = 'auth_return_app'

interface AuthState {
  user: User | null
  session: Session | null
  initialized: boolean
  loading: boolean

  // Sign-in modal visibility (any component can open it via openSignInModal()).
  signInModalOpen: boolean
  openSignInModal: () => void
  closeSignInModal: () => void

  // Bootstrap: read existing session from cookies, subscribe to auth state changes.
  // Call once from App.tsx on mount.
  init: () => Promise<void>

  // OAuth redirect flow (Google, GitHub, and later Web3 via separate methods).
  signInWithProvider: (provider: OAuthProvider) => Promise<{ error: string | null }>

  // Email/password (Supabase native).
  signInWithEmail: (email: string, password: string) => Promise<{ error: string | null }>
  signUpWithEmail: (email: string, password: string) => Promise<{ error: string | null }>

  // Magic link (passwordless email).
  signInWithMagicLink: (email: string) => Promise<{ error: string | null }>

  signOut: () => Promise<void>
}

// Post-auth landing URL.
//
// Production: route through the central auth-router at project-you.app/.
// The router reads `auth_return_app` from localStorage and 301s to the right
// app's path (/vps/app/, /membot/app/, etc.) while preserving ?code= / #access_token.
// Dev: skip the router (no router running locally); land back on the app's own
// origin + BASE_URL.
function redirectTo(): string {
  if (import.meta.env.PROD) {
    return window.location.origin + '/'
  }
  return window.location.origin + (import.meta.env.BASE_URL ?? '/')
}

// Stamp our app identity into localStorage so the central router knows where
// to send the user post-OAuth. Same-origin localStorage survives the roundtrip
// because /vps/app/ and / share origin project-you.app. Silently no-ops if
// localStorage is unavailable (private mode, storage quota hit, etc.).
function stampReturnApp(): void {
  try { localStorage.setItem(RETURN_APP_KEY, APP_ID) } catch { /* ignore */ }
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  session: null,
  initialized: false,
  loading: false,

  signInModalOpen: false,
  openSignInModal: () => set({ signInModalOpen: true }),
  closeSignInModal: () => set({ signInModalOpen: false }),

  init: async () => {
    set({ loading: true })
    const { data } = await supabase.auth.getSession()
    set({
      session: data.session,
      user: data.session?.user ?? null,
      initialized: true,
      loading: false,
    })

    // Stamp app usage if we have ANY session on init. Covers two cases that
    // SIGNED_IN-only handling misses:
    //   1. Post-OAuth code-exchange landed before this listener registered
    //      (@supabase/ssr exchanges synchronously at client construction).
    //   2. Page refresh restored a session from cookies — fires INITIAL_SESSION,
    //      not SIGNED_IN.
    // The track_app_usage RPC is idempotent (its WHERE clause skips when the
    // app is already in apps_list), so calling on every init is safe.
    if (data.session) {
      supabase.rpc('track_app_usage', { app_name: APP_ID }).then(({ error }) => {
        if (error) console.warn('[auth] track_app_usage failed:', error.message)
      })
    }

    supabase.auth.onAuthStateChange((event, session) => {
      set({ session, user: session?.user ?? null })

      // Also re-stamp on SIGNED_IN events that happen post-init (e.g. user
      // signs out + signs back in without a page reload).
      if (event === 'SIGNED_IN' && session) {
        supabase.rpc('track_app_usage', { app_name: APP_ID }).then(({ error }) => {
          if (error) console.warn('[auth] track_app_usage failed:', error.message)
        })
      }
    })
  },

  signInWithProvider: async (provider) => {
    stampReturnApp()
    const { error } = await supabase.auth.signInWithOAuth({
      provider,
      options: { redirectTo: redirectTo() },
    })
    return { error: error?.message ?? null }
  },

  signInWithEmail: async (email, password) => {
    set({ loading: true })
    const { error } = await supabase.auth.signInWithPassword({ email, password })
    set({ loading: false })
    return { error: error?.message ?? null }
  },

  signUpWithEmail: async (email, password) => {
    stampReturnApp()
    set({ loading: true })
    const { error } = await supabase.auth.signUp({
      email,
      password,
      options: { emailRedirectTo: redirectTo() },
    })
    set({ loading: false })
    return { error: error?.message ?? null }
  },

  signInWithMagicLink: async (email) => {
    stampReturnApp()
    set({ loading: true })
    const { error } = await supabase.auth.signInWithOtp({
      email,
      options: { emailRedirectTo: redirectTo() },
    })
    set({ loading: false })
    return { error: error?.message ?? null }
  },

  signOut: async () => {
    await supabase.auth.signOut()
    set({ user: null, session: null })
  },
}))
