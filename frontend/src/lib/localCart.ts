/**
 * Local-cart helpers. Wraps the public/lattice-webgpu/npz-loader.js module
 * for use from React/TS, and implements client-side cosine search over a
 * parsed cart's embeddings + passages so the user can search their own
 * file without uploading anything.
 */

import type { LocalCart } from '../store/appStore';
import type { SearchResult } from '../api/types';

interface ParsedCart {
    embeddings: Float32Array;
    embeddingsShape: number[];
    passages: string[];
    figures?: Map<string, Uint8Array>;
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
        sizeBytes: file.size,
        mountedAt: performance.now(),
        figures: cart.figures ?? new Map(),
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
    for (let i = 0; i < n; i++) {
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
            // Sequential PREV/NEXT — clamped to cart bounds. The fancier
            // hippocampus-aware navigation (which respects document boundaries)
            // would require parsing hippocampus.npy from the cart; sequential
            // by idx is a clean MVP that works for every cart.
            prev_idx: idx > 0 ? idx - 1 : null,
            next_idx: idx < total - 1 ? idx + 1 : null,
        } as SearchResult;
    });
}
