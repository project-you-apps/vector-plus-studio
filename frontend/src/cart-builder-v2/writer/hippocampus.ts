import type { Section } from '../types'

// 64-byte per-pattern hippocampus row. Little-endian throughout.
// Mirrors api/cartridge_io.py — TWO wire formats coexist, distinguished by
// the format_version byte at offset 4:
//
// V1/V2 layout ('<I B B I I I I H I B B 34s'):
//   pattern_id     uint32  4  @0       1-based; 0 = none
//   format_version uint8   1  @4       1 or 2
//   cartridge_type uint8   1  @5
//   parent_ptr     uint32  4  @6       1-based prev; 0 = none
//   child_ptr      uint32  4  @10      1-based next; 0 = none
//   sibling_ptr    uint32  4  @14
//   source_hash    uint32  4  @18      FNV-1a of source filename
//   sequence_num   uint16  2  @22
//   timestamp      uint32  4  @24      Unix seconds
//   flags          uint8   1  @28      tombstone/pinned/perish
//   perms_byte     uint8   1  @29      RWX (v2 only; v1 leaves 0)
//   reserved       34s     34 @30
//
// V3 provenance layout ('<I B B I I I H H H I B B 34s') — 2026-07-18:
//   ...same through byte 17...
//   source_idx     uint16  2  @18      index into top-level source_strings.npy
//   reserved       uint16  2  @20      zero-filled
//   sequence_num   uint16  2  @22      unchanged
//   ...rest identical to V1/V2...
//
// v3 also emits a deduplicated ``source_strings.npy`` unicode array in the
// cart zip. Regulated-industry pilots (legal / clinical / CPA / government)
// require this for auditable per-result provenance. See
// CC_cart-provenance-schema_2026-06-15 for rationale.

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

// Format versions — must match api/cartridge_io.py FORMAT_VERSION_*.
export const FORMAT_VERSION_LEGACY     = 1
export const FORMAT_VERSION_CANONICAL  = 2
export const FORMAT_VERSION_PROVENANCE = 3
export const FORMAT_VERSION_DEFAULT    = FORMAT_VERSION_PROVENANCE

export interface HippocampusOptions {
  format_version?: number
  cartridge_type?: number
  permsByte?: number
  /** Override the timestamp (Unix seconds). Default: now. */
  timestamp?: number
}

export interface PackedHippocampus {
  /** Contiguous ``[N * 64]`` uint8 bytes; reshape to [N, 64] on save. */
  rows: Uint8Array
  /** v3 provenance: deduplicated source strings, ``source_idx`` indexes here.
   *  Null for v1/v2 (writer uses source_hash instead). */
  sourceStrings: string[] | null
}

/**
 * Pack a list of sections into hippocampus rows plus (for v3) a deduplicated
 * source-strings table. Output is contiguous so callers can write ``rows``
 * via npyjs.dump as a single ``hippocampus.npy`` entry inside the cart ZIP,
 * and ``sourceStrings`` as ``source_strings.npy`` alongside.
 */
export function packHippocampus(
  sections: Section[],
  options: HippocampusOptions = {}
): PackedHippocampus {
  const {
    format_version = FORMAT_VERSION_DEFAULT,
    cartridge_type = 0,
    permsByte = PERM_BROWSER_DEFAULT,
    timestamp = Math.floor(Date.now() / 1000),
  } = options

  const emitProvenance = format_version === FORMAT_VERSION_PROVENANCE

  const n = sections.length
  const buffer = new ArrayBuffer(n * HIPPO_SIZE)
  const view = new DataView(buffer)

  // Build the deduplicated source-strings table up front — index 0 is
  // reserved for "no source" (Pattern 0 header + any future headerless).
  let sourceStrings: string[] | null = null
  const sourceIdxByName = new Map<string, number>()
  if (emitProvenance) {
    sourceStrings = ['']
    for (const section of sections) {
      if (!sourceIdxByName.has(section.source)) {
        sourceIdxByName.set(section.source, sourceStrings.length)
        sourceStrings.push(section.source)
      }
    }
    if (sourceStrings.length > 0xffff) {
      throw new Error(
        `source_strings table has ${sourceStrings.length} entries; ` +
        `exceeds uint16 source_idx cap (65535).`
      )
    }
  }

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

    if (emitProvenance) {
      // source_idx uint16 (byte 18-19) + reserved uint16 (byte 20-21)
      const idx = sourceIdxByName.get(section.source) ?? 0
      view.setUint16(offset + 18, idx & 0xffff, true)
      view.setUint16(offset + 20, 0, true)
    } else {
      // source_hash uint32 (byte 18-21) — FNV-1a of source filename
      view.setUint32(offset + 18, fnv1a32(section.source), true)
    }

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

  return {
    rows: new Uint8Array(buffer),
    sourceStrings,
  }
}

function fnv1a32(s: string): number {
  let h = 0x811c9dc5
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return h >>> 0
}
