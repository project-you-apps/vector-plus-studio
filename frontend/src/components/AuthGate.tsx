import type { ReactNode } from 'react'
import { useAuthStore } from '../store/authStore'

interface Props {
  signedIn?: ReactNode
  signedOut?: ReactNode
  loading?: ReactNode
  children?: ReactNode  // shorthand: render only when signed in (same as `signedIn`)
}

// Renders different content based on auth state. Three branches:
//   - not yet initialized (cookie still being read) -> `loading`
//   - signed in -> `signedIn` (or `children` if signedIn is omitted)
//   - signed out -> `signedOut`
//
// Anything omitted renders nothing. Pass nothing at all and you get nothing —
// callers should always specify at least one branch.
export default function AuthGate({ signedIn, signedOut, loading, children }: Props) {
  const initialized = useAuthStore((s) => s.initialized)
  const user = useAuthStore((s) => s.user)

  if (!initialized) return <>{loading ?? null}</>
  if (user) return <>{signedIn ?? children ?? null}</>
  return <>{signedOut ?? null}</>
}
