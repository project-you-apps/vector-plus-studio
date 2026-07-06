/**
 * Local-cart helpers. Wraps the public/lattice-webgpu/npz-loader.js module
 * for use from React/TS, and implements client-side cosine search over a
 * parsed cart's embeddings + passages so the user can search their own
 * file without uploading anything.
 */

import type { LocalCart, LocalCartPattern0Meta, LocalCartPatternMeta } from '../store/appStore';
import type { SearchResult } from '../api/types';

interface ParsedCart {
    embeddings: Float32Array;
    embeddingsShape: number[];
    passages: string[];
    // Provenance v1 sidecar — present iff the cart's .npz contains
    // source_paths.npy (browser-built carts 2026-06-15+). Null/undefined
    // for legacy server-built carts; ResultCard renders the source line
    // only when this is present. See CC_cart-provenance-schema_2026-06-15.
    sourcePaths?: string[] | null;
    figures?: Map<string, Uint8Array>;
    // Pattern-0 metadata parsed from pattern0.npy (single-element unicode
    // NPY holding the same JSON payload as api/cartbuilder/builder.py). Null
    // for legacy carts that predate the sidecar.
    pattern0?: LocalCartPattern0Meta | null;
    // Per-pattern metadata sidecar parsed from per_pattern_meta.npy.
    // One record per pattern in `passages` (parallel-indexed). Null for
    // legacy carts. Andy 2026-07-05 PM: image_b64 for graphic patterns
    // flows through here to the UI thumbnail rendering.
    perPatternMeta?: LocalCartPatternMeta[] | null;
}

let parseCartNpz: ((buffer: ArrayBuffer) => Promise<ParsedCart>) | null = null;

async function loadParser(): Promise<typeof parseCartNpz extends null ? never : NonNullable<typeof parseCartNpz>> {
    if (parseCartNpz) return parseCartNpz;
    const base = import.meta.env.BASE_URL || '/';
    const mod = await import(/* @vite-ignore */ `${base}lattice-webgpu/npz-loader.js`);
    parseCartNpz = mod.parseCartNpz;
    return parseCartNpz!;
}

/**
 * Parse a `.cart.npz` File and return a ready-to-mount LocalCart.
 * Throws with a user-friendly message on any failure.
 */
export async function parseCartFile(file: File): Promise<LocalCart> {
    const parser = await loadParser();
    const buf = await file.arrayBuffer();
    const cart = await parser(buf);

    if (cart.passages.length !== cart.embeddingsShape[0]) {
        throw new Error(
            `Cart structure inconsistent: ${cart.embeddingsShape[0]} embeddings but ${cart.passages.length} passages parsed.`,
        );
    }

    const name = file.name.replace(/\.cart\.npz$/i, '').replace(/\.npz$/i, '');
    return {
        name,
        filename: file.name,
        embeddings: cart.embeddings,
        embeddingsShape: cart.embeddingsShape,
        passages: cart.passages,
        sourcePaths: cart.sourcePaths ?? null,
        sizeBytes: file.size,
        mountedAt: performance.now(),
        figures: cart.figures ?? new Map(),
        pattern0Meta: cart.pattern0 ?? null,
        perPatternMeta: cart.perPatternMeta ?? null,
        // Editable-state defaults: no tombstones, not dirty. Edit Carts mutates
        // these via the localCart* store actions; cosineSearchLocal filters
        // tombstoned idx out of search results.
        tombstones: new Set<number>(),
        dirty: false,
    };
}

function l2NormalizeInPlace(v: Float32Array, offset: number, len: number): number {
    let sumSq = 0;
    for (let i = 0; i < len; i++) {
        const x = v[offset + i];
        sumSq += x * x;
    }
    return Math.sqrt(sumSq) + 1e-9;
}

/**
 * Save a LocalCart back to disk via the cart-builder-v2 writer pipeline.
 * Filters tombstoned passages, rebuilds embeddings + sourcePaths arrays
 * for the surviving entries, then re-bundles as a .cart.npz (+ manifest
 * + permissions) and triggers showDirectoryPicker for the user to pick
 * where to save. Andy 2026-06-16 PM: this is the persistence path for
 * Edit Carts working on browser-mounted LocalCarts on the public droplet.
 *
 * Saved cart is FUNCTIONALLY EQUIVALENT to the source (same embedding dim,
 * same NPY format, same manifest shape) but with tombstoned passages
 * permanently removed from passages.npy / embeddings.npy / source_paths.npy.
 * User can re-mount the saved file to verify, or distribute to others.
 *
 * Returns success+message; the message is the saved filename on success
 * or the error reason on failure (user-cancel counts as success with a
 * cancelled message so UX doesn't show a scary error).
 */
export async function saveLocalCartToDisk(cart: import('../store/appStore').LocalCart): Promise<{ success: boolean; message: string }> {
    const { buildCart, downloadBuiltCart } = await import('../cart-builder-v2/writer/npz');
    const [n, dim] = cart.embeddingsShape;
    // Build the live (non-tombstoned) index list, preserving original order
    // so the cart's narrative integrity is preserved on re-mount.
    const liveIndices: number[] = [];
    for (let i = 0; i < n; i++) {
        if (!cart.tombstones.has(i)) liveIndices.push(i);
    }
    if (liveIndices.length === 0) {
        return { success: false, message: 'Cannot save an empty cart (every passage tombstoned).' };
    }
    // Rebuild embeddings + sections from live entries.
    const liveEmbeddings = new Float32Array(liveIndices.length * dim);
    const liveSections: Array<import('../cart-builder-v2/types').Section> = [];
    for (let outI = 0; outI < liveIndices.length; outI++) {
        const inI = liveIndices[outI];
        const inOff = inI * dim;
        const outOff = outI * dim;
        for (let d = 0; d < dim; d++) liveEmbeddings[outOff + d] = cart.embeddings[inOff + d];
        liveSections.push({
            text: cart.passages[inI] ?? '',
            page: null,
            source: cart.sourcePaths?.[inI] ?? '<unknown>',
        });
    }
    try {
        const built = await buildCart(liveEmbeddings, liveSections, {
            cartName: cart.name,
        });
        await downloadBuiltCart(built);
        return {
            success: true,
            message: `Saved ${built.cartFilename} (${liveSections.length} passages; ${cart.tombstones.size} tombstoned)`,
        };
    } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        // User-cancel of the directory picker shows up as an AbortError -- treat
        // as success-with-cancel so we don't show a scary failure toast.
        if (msg.includes('aborted') || msg.includes('AbortError')) {
            return { success: true, message: 'Save cancelled' };
        }
        return { success: false, message: `Save failed: ${msg}` };
    }
}

/**
 * Browser-side cosine search over a LocalCart. Matches the shape of
 * api/search.py's cosine_search for compatibility with ResultCard rendering.
 *
 * Implementation: row-wise dot product of normalized query against each
 * normalized embedding row. For 100k × 768 carts this is ~76 M multiplies
 * per query, completes in 100-300 ms on a typical laptop without WebGPU.
 */
export function cosineSearchLocal(
    queryEmb: Float32Array,
    cart: LocalCart,
    topK: number,
): SearchResult[] {
    const [n, dim] = cart.embeddingsShape;
    const emb = cart.embeddings;

    // Normalize query.
    const qNorm = l2NormalizeInPlace(queryEmb, 0, dim);
    const qNormalized = new Float32Array(dim);
    for (let d = 0; d < dim; d++) qNormalized[d] = queryEmb[d] / qNorm;

    const scores = new Float32Array(n);
    const tombstones = cart.tombstones;
    for (let i = 0; i < n; i++) {
        // Tombstoned passages get score = -Infinity so they fall out of top-K
        // ranking and never appear in search results. Edit Carts can restore
        // them; until then, they're invisible to search.
        if (tombstones && tombstones.has(i)) {
            scores[i] = -Infinity;
            continue;
        }
        const rowStart = i * dim;
        let dot = 0;
        let rowSumSq = 0;
        for (let d = 0; d < dim; d++) {
            const e = emb[rowStart + d];
            dot += qNormalized[d] * e;
            rowSumSq += e * e;
        }
        const rowNorm = Math.sqrt(rowSumSq) + 1e-9;
        scores[i] = dot / rowNorm;
    }

    // Top-K via single pass with a small heap of K worst-scores.
    const candidateIdx: number[] = [];
    for (let i = 0; i < n; i++) {
        if (candidateIdx.length < topK) {
            candidateIdx.push(i);
            if (candidateIdx.length === topK) {
                candidateIdx.sort((a, b) => scores[b] - scores[a]);
            }
        } else if (scores[i] > scores[candidateIdx[topK - 1]]) {
            candidateIdx[topK - 1] = i;
            // Bubble up.
            for (let j = topK - 1; j > 0 && scores[candidateIdx[j]] > scores[candidateIdx[j - 1]]; j--) {
                const tmp = candidateIdx[j];
                candidateIdx[j] = candidateIdx[j - 1];
                candidateIdx[j - 1] = tmp;
            }
        }
    }
    candidateIdx.sort((a, b) => scores[b] - scores[a]);

    const total = cart.passages.length;
    return candidateIdx.map((idx, rank) => {
        const passage = cart.passages[idx] ?? '';
        const firstNewline = passage.indexOf('\n');
        const title = (firstNewline > 0 ? passage.slice(0, firstNewline) : passage.slice(0, 80)).trim();
        const lines = passage.split('\n');
        const preview = lines.length > 1 ? lines.slice(1, 3).join(' ').slice(0, 200) : '';
        return {
            rank: rank + 1,
            idx,
            score: scores[idx],
            cosine_score: scores[idx],
            physics_score: null,
            hamming_score: null,
            keyword_boost: null,
            title,
            preview,
            full_text: passage,
            from_lattice: false,
            // Provenance v1 sidecar — source filename per result, populated
            // from the cart's source_paths.npy if present. Browser-built carts
            // 2026-06-15+ have this; legacy server-built carts won't, and
            // source_path will be undefined (ResultCard hides the source line).
            source_path: cart.sourcePaths?.[idx] ?? undefined,
            // Sequential PREV/NEXT — clamped to cart bounds. The fancier
            // hippocampus-aware navigation (which respects document boundaries)
            // would require parsing hippocampus.npy from the cart; sequential
            // by idx is a clean MVP that works for every cart.
            prev_idx: idx > 0 ? idx - 1 : null,
            next_idx: idx < total - 1 ? idx + 1 : null,
        } as SearchResult;
    });
}
