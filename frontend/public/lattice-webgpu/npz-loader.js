/**
 * Minimal browser-side .cart.npz / .npz reader.
 *
 * NumPy savez writes a ZIP file with DEFLATE-compressed .npy entries.
 * We use the browser's native DecompressionStream('deflate-raw') so there's
 * no dependency on jszip / fflate / pako.
 *
 * Exports:
 *   parseNpz(buffer)              -> { [name]: { dtype, shape, fortranOrder, data, raw } }
 *   parsePickledStrings(uint8)    -> string[]   // for passages.npy (dtype=object)
 *   parseCartNpz(buffer)          -> { embeddings, embeddingsShape, passages }
 */

import { parseNpy } from './npy-loader.js';

// ---------------------------------------------------------------------------
// ZIP parser
// ---------------------------------------------------------------------------

const ZIP_EOCD_SIG = 0x06054b50;
const ZIP_CD_SIG = 0x02014b50;
const ZIP_LFH_SIG = 0x04034b50;

function findEOCD(view, bytes) {
    const minOffset = Math.max(0, bytes.length - 22 - 65535);
    for (let i = bytes.length - 22; i >= minOffset; i--) {
        if (view.getUint32(i, true) === ZIP_EOCD_SIG) return i;
    }
    throw new Error('NPZ: no EOCD record found (not a valid ZIP file?)');
}

function readZipEntries(buffer) {
    const view = new DataView(buffer);
    const bytes = new Uint8Array(buffer);
    const eocd = findEOCD(view, bytes);
    const entryCount = view.getUint16(eocd + 10, true);
    const cdSize = view.getUint32(eocd + 12, true);
    const cdOffset = view.getUint32(eocd + 16, true);

    const entries = [];
    let p = cdOffset;
    for (let i = 0; i < entryCount; i++) {
        if (view.getUint32(p, true) !== ZIP_CD_SIG) {
            throw new Error(`NPZ: bad central directory signature at ${p}`);
        }
        const compMethod = view.getUint16(p + 10, true);
        const compSize = view.getUint32(p + 20, true);
        const uncompSize = view.getUint32(p + 24, true);
        const nameLen = view.getUint16(p + 28, true);
        const extraLen = view.getUint16(p + 30, true);
        const commentLen = view.getUint16(p + 32, true);
        const localHeaderOff = view.getUint32(p + 42, true);
        const name = new TextDecoder().decode(bytes.subarray(p + 46, p + 46 + nameLen));
        entries.push({ name, compMethod, compSize, uncompSize, localHeaderOff });
        p += 46 + nameLen + extraLen + commentLen;
        if (p > cdOffset + cdSize) throw new Error('NPZ: central directory walk overflowed');
    }
    return entries;
}

async function inflate(compressed) {
    const ds = new DecompressionStream('deflate-raw');
    const stream = new Response(new Blob([compressed]).stream().pipeThrough(ds));
    const ab = await stream.arrayBuffer();
    return new Uint8Array(ab);
}

async function extractEntryBytes(buffer, entry) {
    const view = new DataView(buffer);
    if (view.getUint32(entry.localHeaderOff, true) !== ZIP_LFH_SIG) {
        throw new Error(`NPZ: bad local header signature for ${entry.name}`);
    }
    const nameLen = view.getUint16(entry.localHeaderOff + 26, true);
    const extraLen = view.getUint16(entry.localHeaderOff + 28, true);
    const dataStart = entry.localHeaderOff + 30 + nameLen + extraLen;
    const compressed = new Uint8Array(buffer, dataStart, entry.compSize);
    if (entry.compMethod === 0) return compressed;
    if (entry.compMethod === 8) return await inflate(compressed);
    throw new Error(`NPZ: unsupported compress method ${entry.compMethod} for ${entry.name}`);
}

/**
 * Parse an .npz file into a map of name → parsed NPY result.
 * For object-dtype arrays, parseNpy succeeds in finding the header and dataOffset
 * even though the data itself is a Python pickle blob.
 */
export async function parseNpz(buffer) {
    const entries = readZipEntries(buffer);
    const result = {};
    for (const entry of entries) {
        const raw = await extractEntryBytes(buffer, entry);
        const baseName = entry.name.replace(/\.npy$/i, '');
        try {
            const npy = parseNpy(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
            result[baseName] = { ...npy, raw };
        } catch (err) {
            result[baseName] = { error: err.message, raw };
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Pickle walker — proper opcode table, covers everything numpy emits when
// pickling a 1-D object array of strings.
// Reference: CPython Lib/pickle.py opcode list.
// ---------------------------------------------------------------------------

const STRUCT_NAMES = new Set([
    'numpy', 'numpy.core.multiarray', 'numpy._core.multiarray',
    'numpy.core', 'numpy.dtype', '_reconstruct',
    'ndarray', 'dtype', 'O', 'O8', '|', '|O', 'b', '',
]);

export function parsePickledStrings(bytes) {
    const decoder = new TextDecoder('utf-8');
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const out = [];
    let i = 0;

    while (i < bytes.length) {
        const op = bytes[i];

        switch (op) {
            // No-payload opcodes — advance 1 byte.
            case 0x28: case 0x29: case 0x2e: case 0x30: case 0x31: case 0x32:
            case 0x4e: case 0x51: case 0x52: case 0x5d: case 0x61: case 0x62:
            case 0x64: case 0x65: case 0x6c: case 0x6d: case 0x73: case 0x74:
            case 0x75: case 0x7d: case 0x81: case 0x85: case 0x86: case 0x87:
            case 0x88: case 0x89: case 0x92: case 0x93: case 0x94: case 0x97:
            case 0x98:
                i += 1;
                break;

            // 1-byte payload.
            case 0x4b: case 0x68: case 0x71: case 0x80: case 0x82:
                i += 2;
                break;

            // 2-byte payload.
            case 0x4d: case 0x83:
                i += 3;
                break;

            // 4-byte payload.
            case 0x4a: case 0x6a: case 0x72: case 0x84:
                i += 5;
                break;

            // 8-byte payload (FRAME length is informational — frame contents
            // continue inline, so we just step past the 8 size bytes).
            case 0x47: case 0x95:
                i += 9;
                break;

            // 1-byte LENGTH then LENGTH bytes (non-string bytes payloads).
            case 0x43: case 0x55: case 0x8a: {
                const len = bytes[i + 1];
                i += 2 + len;
                break;
            }

            // 4-byte LENGTH then LENGTH bytes (non-string bytes payloads).
            case 0x42: case 0x54: case 0x8b: {
                const len = view.getUint32(i + 1, true);
                i += 5 + len;
                break;
            }

            // 8-byte LENGTH then LENGTH bytes (non-string bytes payloads).
            case 0x8e: case 0x96: {
                const lo = view.getUint32(i + 1, true);
                const hi = view.getUint32(i + 5, true);
                const len = lo + hi * 0x100000000;
                i += 9 + len;
                break;
            }

            // SHORT_BINUNICODE — 1-byte length + UTF-8 bytes. Collect.
            case 0x8c: {
                const len = bytes[i + 1];
                const str = decoder.decode(bytes.subarray(i + 2, i + 2 + len));
                if (!STRUCT_NAMES.has(str)) out.push(str);
                i += 2 + len;
                break;
            }

            // BINUNICODE — 4-byte length + UTF-8 bytes. Collect.
            case 0x58: {
                const len = view.getUint32(i + 1, true);
                const str = decoder.decode(bytes.subarray(i + 5, i + 5 + len));
                if (!STRUCT_NAMES.has(str)) out.push(str);
                i += 5 + len;
                break;
            }

            // BINUNICODE8 — 8-byte length + UTF-8 bytes. Collect.
            case 0x8d: {
                const lo = view.getUint32(i + 1, true);
                const hi = view.getUint32(i + 5, true);
                const len = lo + hi * 0x100000000;
                const str = decoder.decode(bytes.subarray(i + 9, i + 9 + len));
                if (!STRUCT_NAMES.has(str)) out.push(str);
                i += 9 + len;
                break;
            }

            // Newline-terminated text opcodes (rare in modern protocol 4/5).
            case 0x46: case 0x49: case 0x4c: case 0x50:
            case 0x53: case 0x56: case 0x66: case 0x70: {
                let j = i + 1;
                while (j < bytes.length && bytes[j] !== 0x0a) j += 1;
                i = j + 1;
                break;
            }
            case 0x63: { // GLOBAL has TWO newline-terminated strings.
                let j = i + 1;
                for (let k = 0; k < 2; k++) {
                    while (j < bytes.length && bytes[j] !== 0x0a) j += 1;
                    j += 1;
                }
                i = j;
                break;
            }

            default:
                throw new Error(
                    `Pickle walker: unknown opcode 0x${op.toString(16).padStart(2, '0')} at offset ${i}`,
                );
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Unicode NPY string decoder — handles browser-built carts (cart-builder-v2)
// ---------------------------------------------------------------------------
// numpy dtype '<U<n>' stores each string as `n` UTF-32 LE codepoints (4 bytes
// each), padded with trailing zero codepoints. Layout is row-major over the
// shape. We only need shape=[N] (1-D string array of passages) here.
//
// Total bytes per entry = `maxLen * 4`. Total payload = `N * maxLen * 4`.

/**
 * Decode a unicode NPY string array payload into JS strings.
 * @param {ArrayBuffer} data - raw payload (post-header, post-shape)
 * @param {number} maxLen - codepoints per string (from dtype <U<n>)
 * @param {number[]} shape - NPY shape; we use shape[0] for count
 * @returns {string[]}
 */
export function parseUnicodeStrings(data, maxLen, shape) {
    // Shape [] (0-D) means a numpy scalar — semantically one element, not zero.
    // Passages usually come in as shape [N] (1-D); pattern0 comes in as shape []
    // (0-D scalar of dtype <U<N>). Before this fix, 0-D fell through to count=0
    // and pattern0 was silently dropped despite parsing everything else fine.
    // 2026-07-04: bug found while diagnosing Day 2 pattern0Meta: null in LocalCart mounts.
    const count = shape && shape.length > 0 ? shape[0] : 1;
    const view = new DataView(data);
    const out = new Array(count);
    const bytesPerStr = maxLen * 4;
    for (let i = 0; i < count; i++) {
        const start = i * bytesPerStr;
        // Read codepoints until trailing zero padding kicks in.
        let s = '';
        for (let j = 0; j < maxLen; j++) {
            const off = start + j * 4;
            if (off + 4 > data.byteLength) break;
            const cp = view.getUint32(off, true);
            if (cp === 0) break;
            // String.fromCodePoint handles surrogate-pair-bearing codepoints
            // (e.g. emoji) correctly, unlike fromCharCode.
            s += String.fromCodePoint(cp);
        }
        out[i] = s;
    }
    return out;
}

// ---------------------------------------------------------------------------
// High-level: parse a .cart.npz and return embeddings + passages
// ---------------------------------------------------------------------------

export async function parseCartNpz(buffer) {
    // Walk the ZIP twice: once for .npy entries (handled by parseNpz already),
    // once for "figures/<name>" entries which are raw image bytes embedded by
    // the artifact ingestor's post-process step. Numpy savez ignores them on
    // the Python side; we read them here for browser-side rendering.
    const npzEntries = await parseNpz(buffer);

    const embEntry = npzEntries['embeddings'];
    if (!embEntry || embEntry.error || !embEntry.data) {
        const found = Object.keys(npzEntries).join(', ');
        throw new Error(
            `Cart: embeddings.npy missing or unparseable. File contains: [${found}]. ` +
            `This may be a signatures file (.npz) rather than a cart (.cart.npz).`,
        );
    }
    if (embEntry.dtype !== '<f4' && embEntry.dtype !== '|f4' && embEntry.dtype !== 'f4') {
        throw new Error(`Cart: embeddings dtype must be float32, got '${embEntry.dtype}'`);
    }
    const embeddings = new Float32Array(embEntry.data);

    const passagesEntry = npzEntries['passages'];
    if (!passagesEntry || !passagesEntry.data) {
        throw new Error('Cart: passages.npy missing');
    }
    // Passages can be in two formats:
    //   (a) Python-built carts: dtype is object-array, payload is a pickle blob
    //       (parsePickledStrings handles this).
    //   (b) Browser-built carts (cart-builder-v2): dtype is <U<maxlen>, payload
    //       is UTF-32 LE codepoints (4 bytes each), shape=[N], total bytes =
    //       N * maxlen * 4.
    // We pick the path based on the NPY header's dtype string.
    let passages;
    const passagesDtype = passagesEntry.dtype || '';
    const unicodeMatch = passagesDtype.match(/^<U(\d+)$/);
    if (unicodeMatch) {
        const maxLen = parseInt(unicodeMatch[1], 10);
        passages = parseUnicodeStrings(passagesEntry.data, maxLen, passagesEntry.shape);
    } else {
        // Fallback: legacy server-built carts (dtype 'O' / unknown). Try pickle.
        const pickleBytes = new Uint8Array(passagesEntry.data);
        passages = parsePickledStrings(pickleBytes);
    }

    // Provenance resolution — two paths, both produce a per-pattern
    // ``sourcePaths`` array:
    //
    //   v3 (2026-07-18+): source_strings.npy (deduplicated unicode array)
    //   is indexed by each pattern's h-row source_idx uint16 at byte 18.
    //   Preferred — canonical because it can't drift out of sync with h-row.
    //
    //   v1 sidecar (2026-06-15+): source_paths.npy (per-pattern unicode
    //   array). Fallback for pre-v3 carts and safety net during transition.
    //
    // Older carts (server-built or pre-2026-06-15) have neither; sourcePaths
    // stays null and result cards just don't render a source line.
    let sourcePaths = null;

    // Try v3 provenance first — needs source_strings + h-row source_idx.
    const sourceStringsEntry = npzEntries['source_strings'];
    const hippocampusEntry = npzEntries['hippocampus'];
    if (
        sourceStringsEntry && sourceStringsEntry.data &&
        hippocampusEntry && hippocampusEntry.data
    ) {
        const ssDtype = sourceStringsEntry.dtype || '';
        const ssMatch = ssDtype.match(/^<U(\d+)$/);
        if (ssMatch) {
            const ssMaxLen = parseInt(ssMatch[1], 10);
            const sourceStrings = parseUnicodeStrings(
                sourceStringsEntry.data, ssMaxLen, sourceStringsEntry.shape,
            );
            // hippocampus is a [N, 64] uint8 array. Peek row 0 byte 4 to
            // confirm format_version = 3; then resolve each pattern's
            // source_idx (uint16 LE at bytes 18-19) into the strings table.
            const hippoBytes = new Uint8Array(hippocampusEntry.data);
            const nRows = Math.floor(hippoBytes.length / 64);
            if (nRows > 0 && hippoBytes[4] === 3) {  // FORMAT_VERSION_PROVENANCE
                sourcePaths = new Array(nRows);
                const hippoView = new DataView(hippoBytes.buffer, hippoBytes.byteOffset, hippoBytes.byteLength);
                for (let i = 0; i < nRows; i++) {
                    const idx = hippoView.getUint16(i * 64 + 18, true);
                    sourcePaths[i] = (idx >= 0 && idx < sourceStrings.length)
                        ? sourceStrings[idx]
                        : '';
                }
            }
        }
    }

    // Fall back to v1 sidecar if v3 resolution didn't produce a table.
    if (sourcePaths === null) {
        const sourcePathsEntry = npzEntries['source_paths'];
        if (sourcePathsEntry && sourcePathsEntry.data) {
            const spDtype = sourcePathsEntry.dtype || '';
            const spMatch = spDtype.match(/^<U(\d+)$/);
            if (spMatch) {
                const spMaxLen = parseInt(spMatch[1], 10);
                sourcePaths = parseUnicodeStrings(sourcePathsEntry.data, spMaxLen, sourcePathsEntry.shape);
            }
        }
    }

    // Pattern-0 metadata — single-element unicode NPY of a JSON payload
    // (see cart-builder-v2/writer/npz.ts:pattern0_data). Only present for
    // browser-built carts 2026-07-02+ (and for server-built carts that ran
    // through the equivalent writer). Older carts leave pattern0 = null and
    // the Pattern-0 TOC panel falls back to derived stats.
    let pattern0 = null;
    const pattern0Entry = npzEntries['pattern0'];
    // 2026-07-04 debug: expose what's actually in the pattern0 entry when the
    // parse would fail, so we can see WHY. Remove once LocalCart Pattern-0 is
    // fully verified across all cart-build paths.
    if (!pattern0Entry) {
        console.warn('[npz-loader] pattern0 entry MISSING from npzEntries. Keys present:', Object.keys(npzEntries));
    } else if (!pattern0Entry.data) {
        console.warn('[npz-loader] pattern0 entry present but has no .data field. Entry:', pattern0Entry);
    }
    if (pattern0Entry && pattern0Entry.data) {
        const p0Dtype = pattern0Entry.dtype || '';
        const p0Match = p0Dtype.match(/^<U(\d+)$/);
        // Try pickle fallback (like passages does) if dtype isn't the unicode
        // form. Python-side savez sometimes stores a Python str as an object
        // array (dtype 'O') which we'd otherwise silently drop.
        if (!p0Match) {
            console.warn(
                `[npz-loader] pattern0 dtype '${p0Dtype}' didn't match <U<N>. `
                + `Trying pickle fallback. Entry shape=${JSON.stringify(pattern0Entry.shape)}.`
            );
            try {
                const pickleBytes = new Uint8Array(pattern0Entry.data);
                const asStrings = parsePickledStrings(pickleBytes);
                if (asStrings.length > 0 && asStrings[0]) {
                    try {
                        pattern0 = JSON.parse(asStrings[0]);
                    } catch (e) {
                        console.warn('[npz-loader] pickle-parsed pattern0 string is not JSON:', e);
                    }
                }
            } catch (e) {
                console.warn('[npz-loader] pickle fallback for pattern0 failed:', e);
            }
        } else {
            const p0MaxLen = parseInt(p0Match[1], 10);
            const decoded = parseUnicodeStrings(pattern0Entry.data, p0MaxLen, pattern0Entry.shape);
            if (decoded.length > 0 && decoded[0]) {
                try {
                    pattern0 = JSON.parse(decoded[0]);
                } catch {
                    // Malformed JSON — treat as missing rather than crash the mount.
                    pattern0 = null;
                    console.warn('[npz-loader] pattern0 unicode-decoded but not valid JSON');
                }
            }
        }
    }

    // Second pass: figures/* entries. parseNpz already inflated them as
    // "raw" Uint8Array (since they don't look like .npy files, parseNpy errors
    // out and we keep the raw bytes). Walk the entry list to collect them
    // keyed by basename so the UI can look them up by hash without the prefix.
    const figures = new Map();
    const allEntries = readZipEntries(buffer);
    for (const e of allEntries) {
        if (!e.name.startsWith('figures/')) continue;
        const bytes = await extractEntryBytes(buffer, e);
        // basename without 'figures/' prefix — e.g. 'abc123.png'
        const basename = e.name.slice('figures/'.length);
        figures.set(basename, bytes);
    }

    // per_pattern_meta.npy — Day 2 sidecar carrying content_type + per-pattern
    // extras (image_b64 for graphics, html for tables, bbox, caption, ...).
    // Same unicode-array-of-JSON-string shape as pattern0.npy.
    // PM: read this so LocalCart mounts can surface graphic thumbnails in the
    // UI (Result cards, PassageModal, TOC drill-down). Zero-cost passthrough
    // when the entry is absent (legacy carts).
    let perPatternMeta = null;
    const ppmEntry = npzEntries['per_pattern_meta'];
    if (ppmEntry && ppmEntry.data) {
        const ppmDtype = ppmEntry.dtype || '';
        const ppmUnicodeMatch = ppmDtype.match(/^<U(\d+)$/);
        if (ppmUnicodeMatch) {
            try {
                const maxLen = parseInt(ppmUnicodeMatch[1], 10);
                const decoded = parseUnicodeStrings(ppmEntry.data, maxLen, ppmEntry.shape);
                if (decoded.length > 0 && decoded[0]) {
                    perPatternMeta = JSON.parse(decoded[0]);
                }
            } catch (e) {
                console.warn('[npz-loader] per_pattern_meta unicode parse failed:', e);
            }
        } else {
            // Backend variant uses np.savez_compressed → object array (pickled).
            try {
                const pickleBytes = new Uint8Array(ppmEntry.data);
                const asStrings = parsePickledStrings(pickleBytes);
                if (asStrings.length > 0 && asStrings[0]) {
                    perPatternMeta = JSON.parse(asStrings[0]);
                }
            } catch (e) {
                console.warn('[npz-loader] per_pattern_meta pickle parse failed:', e);
            }
        }
    }

    return {
        embeddings,
        embeddingsShape: embEntry.shape,
        passages,
        sourcePaths,
        figures,
        pattern0,
        perPatternMeta,
    };
}
