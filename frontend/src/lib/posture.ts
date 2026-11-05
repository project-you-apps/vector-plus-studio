/**
 * Deployment posture — WHERE is this app, and WHERE does its data go?
 *
 * ⭐⭐ THE POINT IS THE TRUST LABEL, NOT THE VALUE. Andy's goal (2026-11-04) is one
 * browser as the front end for both the local and the hosted stack, where a user with
 * PII/HIPAA concerns can see that a local install stayed local. That only means
 * something if each displayed fact says HOW IT IS KNOWN, because the three sources are
 * not equally trustworthy:
 *
 *   'browser' — the user agent enforces it. Our code cannot forge it. `location` is the
 *               only fact here the page cannot lie about.
 *   'server'  — the backend asserted it over /api/status. Trustworthy about the backend
 *               (it is the only party that knows where the backend is), and worthless if
 *               the backend is not the one you think you are talking to.
 *   'bundle'  — baked in at build time by vite.config.ts. Says what was true WHEN BUILT.
 *
 * ⛔ WHAT THIS DELIBERATELY DOES NOT CLAIM. It is not a data-residency guarantee and
 * must never be worded as one:
 *   1. The browser cannot see through the dev `/api` proxy (see vite.config.ts). A page
 *      on localhost can be driving a public backend.
 *   2. "Local backend" says nothing about the BACKEND's own egress -- Supabase auth,
 *      model downloads, and the membot droplet are all still reachable from it.
 *   3. ⚠ Even on the local-cart path the QUERY leaves the browser: appStore POSTs it to
 *      /api/embed to get an embedding. "The user's file never leaves the browser" is
 *      true of PASSAGE BODIES and false of query text -- and for a HIPAA user a query
 *      ("what is Smith's diagnosis") is itself PHI. A local backend is what makes that
 *      acceptable, which is precisely why the badge has to be honest about reaching one.
 *
 * ⇒ So this answers "is the app and its API on this machine", which is a real and
 * checkable question, and refuses to answer "did any data leave", which is not one the
 * browser can answer. The compliance wording lives in COMPLIANCE-posture.md.
 */

declare const __BUILD_SHA__: string
declare const __BUILD_DIRTY__: boolean
declare const __BUILD_TIME__: string
declare const __BUILD_MODE__: string
declare const __DEV_PROXY_TARGET__: string

export type TrustSource = 'browser' | 'server' | 'bundle'

/** 'local' and 'hosted' are coherent. 'mixed' and 'unknown' are the ones that matter. */
export type PostureVerdict = 'local' | 'hosted' | 'mixed' | 'unknown'

export interface Posture {
  verdict: PostureVerdict
  /** One short phrase for the footer chip. */
  label: string
  /** Why the verdict is what it is, in plain language. Shown on hover and in About. */
  detail: string
  /** Browser-enforced: the origin this document was actually served from. */
  servedFrom: string
  /** Browser-enforced: that origin is a loopback address. */
  servedFromLoopback: boolean
  /** Server-asserted. null until /api/status has answered -- NOT false. */
  backendIsPublic: boolean | null
  buildSha: string
  buildDirty: boolean
  buildTime: string
  buildMode: string
  /** Dev-proxy target, '' in a static build where no proxy exists. */
  devProxyTarget: string
}

/**
 * Loopback test over the HOSTNAME only.
 *
 * ⛔ Not a substring match. `localhost.evil.com` and `127.0.0.1.attacker.net` both
 * CONTAIN a loopback string and are both remote, so a naive `includes()` here would turn
 * the one browser-enforced fact we have into an attacker-controlled one. Exact match, plus
 * the RFC 6761 `.localhost` TLD which resolvers are required to treat as loopback.
 */
export function isLoopbackHostname(hostname: string): boolean {
  const h = hostname.toLowerCase().replace(/^\[|\]$/g, '')
  if (h === 'localhost' || h === '::1' || h === '0:0:0:0:0:0:0:1') return true
  if (h.endsWith('.localhost')) return true
  // 127.0.0.0/8 -- the whole block is loopback, not just 127.0.0.1.
  const m = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(h)
  if (m) {
    const octets = m.slice(1).map(Number)
    if (octets.every((o) => o >= 0 && o <= 255) && octets[0] === 127) return true
  }
  return false
}

export function computePosture(backendIsPublic: boolean | null): Posture {
  const loc = typeof window !== 'undefined' ? window.location : undefined
  const servedFrom = loc ? loc.origin : 'unknown'
  const servedFromLoopback = loc ? isLoopbackHostname(loc.hostname) : false

  let verdict: PostureVerdict
  let label: string
  let detail: string

  if (backendIsPublic === null) {
    // ⛔ An unreached backend is UNKNOWN, never "local". Defaulting to the reassuring
    // answer when the check has not completed is how a safety indicator comes to mean
    // nothing -- it would read green for a server that is simply down.
    verdict = 'unknown'
    label = servedFromLoopback ? 'Local page · backend not reached' : 'Backend not reached'
    detail =
      `This page was served from ${servedFrom}, which the browser enforces. The backend ` +
      `has not answered /api/status yet, so where your data goes is NOT yet established.`
  } else if (servedFromLoopback && !backendIsPublic) {
    verdict = 'local'
    label = 'Local install'
    detail =
      `Served from ${servedFrom} (browser-enforced) and the backend reports it is not a ` +
      `public host. The app and its API are both on this machine. ⚠ This describes ` +
      `LOCATION, not data residency: the backend can still reach the internet on its own.`
  } else if (!servedFromLoopback && backendIsPublic) {
    verdict = 'hosted'
    label = 'Hosted'
    detail =
      `Served from ${servedFrom} and the backend reports it is a public host. Carts you ` +
      `mount from your computer are read in the browser, but queries and uploads go to a ` +
      `server you do not control.`
  } else if (servedFromLoopback && backendIsPublic) {
    // ⭐⭐ THE CASE THIS WHOLE MODULE EXISTS FOR, and the one a frontend-only badge would
    // have shown as "local". The page is on your machine; the data is going elsewhere.
    verdict = 'mixed'
    label = 'MIXED — local page, public backend'
    detail =
      `⛔ This page came from ${servedFrom} on your machine, but the backend answering it ` +
      `reports that it is a PUBLIC host. Query text and any upload are leaving this ` +
      `machine. A local-looking address bar does not mean local data.`
  } else {
    // Served from a remote origin, backend says it is not public: an internal/LAN
    // deployment, or a misconfiguration. Either way, not a claim we should dress up.
    verdict = 'mixed'
    label = 'Remote page, non-public backend'
    detail =
      `Served from ${servedFrom}, which is not a loopback address, while the backend ` +
      `reports it is not a public host. That combination usually means a LAN or ` +
      `reverse-proxied deployment -- treat it as remote unless you set it up yourself.`
  }

  return {
    verdict,
    label,
    detail,
    servedFrom,
    servedFromLoopback,
    backendIsPublic,
    buildSha: typeof __BUILD_SHA__ === 'string' ? __BUILD_SHA__ : 'unknown',
    buildDirty: typeof __BUILD_DIRTY__ === 'boolean' ? __BUILD_DIRTY__ : false,
    buildTime: typeof __BUILD_TIME__ === 'string' ? __BUILD_TIME__ : '',
    buildMode: typeof __BUILD_MODE__ === 'string' ? __BUILD_MODE__ : 'unknown',
    devProxyTarget: typeof __DEV_PROXY_TARGET__ === 'string' ? __DEV_PROXY_TARGET__ : '',
  }
}

/** Short build string for the footer: `66d3d09` / `66d3d09+edits` / `unknown`. */
export function buildStamp(p: Posture): string {
  if (p.buildSha === 'unknown') return 'build unknown'
  return p.buildDirty ? `${p.buildSha}+edits` : p.buildSha
}
