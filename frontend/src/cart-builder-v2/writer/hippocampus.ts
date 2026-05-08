import type { Section } from '../types'

// 64-byte per-pattern hippocampus row format. Little-endian throughout.
// Mirrors api/cartridge_io.py HIPPO_FORMAT exactly:
//   '<I B B I I I I H I B B 34s'
//
// Field          Type    Bytes  Offset  Notes
// pattern_id     uint32  4      0       1-based; 0 = none
// format_version uint8   1      4
// cartridge_type uint8   1      5
// parent_ptr     uint32  4      6       1-based prev pattern; 0 = none
// child_ptr      uint32  4      10      1-based next pattern; 0 = none
// sibling_ptr    uint32  4      14      0 for linear carts
// source_hash    uint32  4      18      hash of source filename
// sequence_num   uint16  2      22      seq within source
// timestamp      uint32  4      24      Unix seconds (uint32 — fits until 2106)
// flags          uint8   1      28      membot's tombstone/pinned bits (0 for fresh)
// perms_byte     uint8   1      29      Step 2b RWX (R=0x01 W=0x02 X=0x04)
// reserved       34s     34     30      zero-filled

export const HIPPO_SIZE = 64

// Step 2b perms bits (must match api/cartridge_io.py PERM_*).
export const PERM_R = 0x01
export const PERM_W = 0x02
export const PERM_X = 0x04

// Browser-built default per-pattern bits: explicitly readable + writable.
// The cart-level `permissions.json` sidecar (Step 2a) is what enforces the
// "read-only by default" deployment policy. Per-pattern bits stay permissive
// so users can re-imprint locally if they download the cart.
export const PERM_BROWSER_DEFAULT = PERM_R | PERM_W

export interface HippocampusOptions {
  format_version?: number
  cartridge_type?: number
  permsByte?: number
  /** Override the timestamp (Unix seconds). Default: now. */
  timestamp?: number
}

/**
 * Pack a list of sections into a `[N, 64]` uint8 array of hippocampus rows.
 * Output is contiguous so callers can write it via npyjs.dump as a single
 * `hippocampus.npy` entry inside the cart ZIP.
 */
export function packHippocampus(
  sections: Section[],
  options: HippocampusOptions = {}
): Uint8Array {
  const {
    format_version = 1,
    cartridge_type = 0,
    permsByte = PERM_BROWSER_DEFAULT,
    timestamp = Math.floor(Date.now() / 1000),
  } = options

  const n = sections.length
  const buffer = new ArrayBuffer(n * HIPPO_SIZE)
  const view = new DataView(buffer)

  for (let i = 0; i < n; i++) {
    const offset = i * HIPPO_SIZE
    const section = sections[i]

    // pattern_id (1-based)
    view.setUint32(offset + 0, i + 1, true)
    // format_version
    view.setUint8(offset + 4, format_version & 0xff)
    // cartridge_type
    view.setUint8(offset + 5, cartridge_type & 0xff)
    // parent_ptr (PREV) — 1-based, 0 for first
    view.setUint32(offset + 6, i > 0 ? i : 0, true)
    // child_ptr (NEXT) — 1-based, 0 for last
    view.setUint32(offset + 10, i < n - 1 ? i + 2 : 0, true)
    // sibling_ptr — linear carts have no siblings
    view.setUint32(offset + 14, 0, true)
    // source_hash — FNV-1a of source filename (good enough for a 32-bit field)
    view.setUint32(offset + 18, fnv1a32(section.source), true)
    // sequence_num — best-effort; clamp to uint16
    view.setUint16(offset + 22, Math.min(i, 0xffff), true)
    // timestamp — Unix seconds
    view.setUint32(offset + 24, timestamp >>> 0, true)
    // flags — membot bits, all 0 for browser-built fresh carts
    view.setUint8(offset + 28, 0)
    // perms_byte — Step 2b
    view.setUint8(offset + 29, permsByte & 0xff)
    // reserved bytes 30..63 are already 0 from `new ArrayBuffer`.
  }

  return new Uint8Array(buffer)
}

function fnv1a32(s: string): number {
  let h = 0x811c9dc5
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return h >>> 0
}
