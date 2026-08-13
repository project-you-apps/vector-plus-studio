/**
 * This tab's identity, and the cart this tab is looking at.
 *
 * WHY THE TAB OWNS THIS. Until 2026-08-13 the server remembered one mounted cart for the
 * whole process, and the UI read it back from `/api/status`. That meant Susie mounting
 * Finance and Betty mounting Revenue fought over one variable — and so did two TABS of the
 * same person, because a server-side binding is keyed by seat and two tabs are one seat.
 * Comparing two carts side by side is an ordinary thing to do, so the tab has to carry its
 * own answer.
 *
 * sessionStorage, NOT localStorage — deliberately:
 *   • sessionStorage is per-TAB, so two tabs get two carts. That is the point.
 *   • It dies with the tab, which is exactly the lifetime Andy specified for anonymous
 *     visitors: "a temp ID that exists as long as they have the site loaded."
 *   • localStorage would put every tab back on one cart, i.e. the bug, one scope down.
 *
 * ⚠ THE TAB ID IS NOT AN IDENTITY. It says which cart this tab is viewing and nothing else.
 * The server never lets it influence an access decision — access comes from the verified
 * token, and a forged X-VPS-Session buys exactly one thing: which cart you are looking at.
 * See api/request_cart.py, which keeps `view_key` and `access_seat` deliberately apart.
 *
 * No store dependency, so `client.ts` can read it without a circular import.
 */

const TAB_ID_KEY = 'vps.tab.id'
const VIEWING_CART_KEY = 'vps.tab.cart'

function newId(): string {
    // randomUUID needs a secure context; plain http://<lan-ip> during a demo is not one, and
    // failing there would break the whole app for a bookkeeping string.
    const c = globalThis.crypto as Crypto | undefined
    if (c?.randomUUID) return c.randomUUID()
    if (c?.getRandomValues) {
        const b = new Uint8Array(16)
        c.getRandomValues(b)
        return Array.from(b, (x) => x.toString(16).padStart(2, '0')).join('')
    }
    return `t${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`
}

/** Stable for the life of this tab. Created on first use. */
export function tabSessionId(): string {
    try {
        let id = sessionStorage.getItem(TAB_ID_KEY)
        if (!id) {
            id = newId()
            sessionStorage.setItem(TAB_ID_KEY, id)
        }
        return id
    } catch {
        // Private mode, or storage disabled. A per-page-load id still separates this tab
        // from other browsers; it just does not survive a refresh. Better than throwing.
        return newId()
    }
}

/** The cart this tab is viewing, or null before anything is mounted. */
export function viewingCart(): string | null {
    try {
        return sessionStorage.getItem(VIEWING_CART_KEY)
    } catch {
        return null
    }
}

/**
 * Record which cart this tab is on. Called after a mount SUCCEEDS, never before —
 * claiming a cart we failed to open would send every later request at something the
 * server will refuse.
 */
export function setViewingCart(cartId: string | null): void {
    try {
        if (cartId) sessionStorage.setItem(VIEWING_CART_KEY, cartId)
        else sessionStorage.removeItem(VIEWING_CART_KEY)
    } catch {
        /* storage disabled — requests fall back to the server's cart, i.e. today's behaviour */
    }
}
