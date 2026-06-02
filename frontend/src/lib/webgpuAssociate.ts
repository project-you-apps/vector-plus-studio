/**
 * WebGPU Associate orchestrator — bridges the React app to the lattice engine
 * + shaders + encoders served from frontend/public/lattice-webgpu/.
 *
 * Pattern mirrors frontend/src/cart-builder-v2/embedder/loader.ts:
 *   - 2-stage WebGPU detection (`'gpu' in navigator` then `requestAdapter`)
 *   - Lazy singleton engine + Promise dedup
 *   - Brain cache keyed by cart name (skip refetch on repeat queries)
 *   - Async runAssociate function takes a query string + cart + top_k + pool_size
 *
 * The engine + shader assets live in public/ (served as static files),
 * imported dynamically with @vite-ignore so Vite doesn't try to bundle them
 * (they fetch their own WGSL shaders at runtime via relative paths).
 */

import type { SearchResult } from '../api/types';

const ROW_FULLY_PROTECTED = 0xFF;
const HIPPO_ROW = 63;
const SETTLE_FRAMES = 30;

// Singleton state — module-level caches so React re-renders don't re-init.
let detectionPromise: Promise<boolean> | null = null;
let detectionResult: boolean | null = null;
let enginePromise: Promise<any> | null = null;
let engineInstance: any = null;
const brainPromises: Map<string, Promise<void>> = new Map();
const brainLoaded: Set<string> = new Set();

// Cached imports of the public/ JS modules — populated on first engine init.
let LatticeEngineCtor: any = null;
let abEncodeRegionFill: any = null;
let abDecodeRegionFillAnalog: any = null;
let l2Normalize: any = null;
let dot: any = null;
let parseNpy: any = null;
let fetchNpyWithProgress: any = null;

function assetUrl(path: string): string {
    const base = import.meta.env.BASE_URL || '/';
    return `${base}lattice-webgpu/${path}`;
}

/**
 * 2-stage WebGPU detection. Returns true only if a working adapter exists.
 * Caches result so repeated calls are free.
 */
export async function detectWebGPU(): Promise<boolean> {
    if (detectionResult !== null) return detectionResult;
    if (detectionPromise) return detectionPromise;

    detectionPromise = (async () => {
        const nav = navigator as Navigator & { gpu?: { requestAdapter: () => Promise<unknown> } };
        if (!nav.gpu) {
            detectionResult = false;
            return false;
        }
        try {
            const adapter = await nav.gpu.requestAdapter();
            detectionResult = !!adapter;
            return detectionResult;
        } catch {
            detectionResult = false;
            return false;
        }
    })();
    return detectionPromise;
}

async function loadPublicModules() {
    if (LatticeEngineCtor) return;
    const engineMod = await import(/* @vite-ignore */ assetUrl('lattice-engine.js'));
    const encoderMod = await import(/* @vite-ignore */ assetUrl('region-fill-encoder.js'));
    const npyMod = await import(/* @vite-ignore */ assetUrl('npy-loader.js'));
    LatticeEngineCtor = engineMod.LatticeEngine;
    abEncodeRegionFill = encoderMod.abEncodeRegionFill;
    abDecodeRegionFillAnalog = encoderMod.abDecodeRegionFillAnalog;
    l2Normalize = encoderMod.l2Normalize;
    dot = encoderMod.dot;
    parseNpy = npyMod.parseNpy;
    fetchNpyWithProgress = npyMod.fetchNpyWithProgress;
}

/**
 * Lazy engine singleton. Constructs LatticeEngine(4096), initializes WebGPU,
 * sets quality profile, protects hippocampus row.
 */
async function getEngine(): Promise<any> {
    if (engineInstance) return engineInstance;
    if (enginePromise) return enginePromise;

    enginePromise = (async () => {
        await loadPublicModules();
        const engine = new LatticeEngineCtor(4096);
        await engine.init();
        engine.setProfile('quality');
        engine.setRowPhysics(HIPPO_ROW, ROW_FULLY_PROTECTED);
        engineInstance = engine;
        return engine;
    })();
    return enginePromise;
}

export interface BrainLoadProgress {
    cartName: string;
    loaded: number;
    total: number;
    stage: 'fetching' | 'parsing' | 'uploading' | 'done';
}

/**
 * Load a cart's brain into the WebGPU engine. Idempotent — second call for
 * the same cart returns instantly (the brain is already in GPU memory).
 * Concurrent calls for the same cart dedupe on the in-flight Promise.
 */
export async function loadBrainForCart(
    cartName: string,
    onProgress?: (p: BrainLoadProgress) => void,
): Promise<void> {
    if (brainLoaded.has(cartName)) return;
    const existing = brainPromises.get(cartName);
    if (existing) return existing;

    const promise = (async () => {
        const engine = await getEngine();
        const url = `/api/cartridges/${encodeURIComponent(cartName)}/brain`;

        const npyBuffer = await fetchNpyWithProgress(url, (loaded: number, total: number) => {
            onProgress?.({ cartName, loaded, total, stage: 'fetching' });
        });
        onProgress?.({ cartName, loaded: npyBuffer.byteLength, total: npyBuffer.byteLength, stage: 'parsing' });

        const npy = parseNpy(npyBuffer);
        if (npy.dtype !== '<u4') {
            throw new Error(`Expected dtype '<u4' for brain, got '${npy.dtype}'`);
        }
        const N = engine.numNeurons;
        const total = new Uint32Array(npy.data);
        let state: Uint32Array, weights: Uint32Array;
        if (total.length === N * 2) {
            state = total.subarray(0, N);
            weights = total.subarray(N, N * 2);
        } else if (total.length === N) {
            state = new Uint32Array(N);
            weights = total;
        } else {
            throw new Error(`Unexpected brain size: ${total.length} uint32s for ${cartName}`);
        }

        onProgress?.({ cartName, loaded: npyBuffer.byteLength, total: npyBuffer.byteLength, stage: 'uploading' });
        await engine.loadBrain(state, weights);
        brainLoaded.add(cartName);
        onProgress?.({ cartName, loaded: npyBuffer.byteLength, total: npyBuffer.byteLength, stage: 'done' });
    })();

    brainPromises.set(cartName, promise);
    try {
        await promise;
    } finally {
        brainPromises.delete(cartName);
    }
}

/**
 * Settle a Nomic embedding through the loaded brain's physics and return
 * the unit-normalized analog-decoded vector. Mirrors the per-candidate
 * inner loop of api/search.py:_associate_search_two_sided.
 */
async function settleEmbedding(emb: Float32Array): Promise<Float32Array> {
    const engine = await getEngine();
    const pattern = abEncodeRegionFill(emb);
    await engine.reset(); // state-only zero; weights preserved (Phase 2c fix)
    await engine.imprintPattern(pattern);
    await engine.settle(SETTLE_FRAMES, false);
    const settled = await engine.recall();
    const decoded = abDecodeRegionFillAnalog(settled);
    return l2Normalize(decoded);
}

export interface RunAssociateOptions {
    query: string;
    cartName: string;
    topK: number;
    poolSize?: number; // candidates from server cosine pre-filter (default 50)
    onStatus?: (msg: string) => void;
}

/**
 * Full browser-side two-sided Associate v83. Server contributes:
 *   - the Nomic embedding for the query text (cheap)
 *   - the cosine candidate pool (cheap, vectorized)
 * Browser does the expensive per-candidate physics settle.
 * Returns SearchResult records compatible with the existing UI.
 */
export async function runWebGpuAssociate(opts: RunAssociateOptions): Promise<SearchResult[]> {
    const { query, cartName, topK, poolSize = 50, onStatus } = opts;

    onStatus?.('Ensuring brain is loaded…');
    await loadBrainForCart(cartName);

    onStatus?.('Embedding query…');
    const embResp = await fetch('/api/embed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
    });
    if (!embResp.ok) {
        const err = await embResp.text();
        throw new Error(`Query embed failed: ${embResp.status} — ${err}`);
    }
    const embData = await embResp.json();
    const queryEmb = new Float32Array(embData.embedding);

    onStatus?.('Fetching candidate pool…');
    const candResp = await fetch(`/api/cartridges/${encodeURIComponent(cartName)}/cosine-candidates`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ embedding: Array.from(queryEmb), pool_size: poolSize }),
    });
    if (!candResp.ok) {
        const err = await candResp.text();
        throw new Error(`Candidate pool fetch failed: ${candResp.status} — ${err}`);
    }
    const candData = await candResp.json();

    onStatus?.(`Settling query through physics…`);
    const settledQ = await settleEmbedding(queryEmb);

    type ScoredCandidate = {
        idx: number;
        cosine_score: number;
        physics_score: number;
        passage: string;
    };
    const scored: ScoredCandidate[] = [];
    for (let i = 0; i < candData.candidates.length; i++) {
        const c = candData.candidates[i];
        onStatus?.(`Settling candidate ${i + 1}/${candData.candidates.length}…`);
        const candEmb = new Float32Array(c.embedding);
        const settledC = await settleEmbedding(candEmb);
        const physicsScore = dot(settledQ, settledC);
        scored.push({
            idx: c.idx,
            cosine_score: c.cosine_score,
            physics_score: physicsScore,
            passage: c.passage,
        });
    }
    scored.sort((a, b) => b.physics_score - a.physics_score);

    return scored.slice(0, topK).map((s, rank) => {
        const passage = s.passage ?? '';
        const firstNewline = passage.indexOf('\n');
        const title = firstNewline > 0 ? passage.slice(0, firstNewline) : passage.slice(0, 80);
        return {
            rank: rank + 1,
            idx: s.idx,
            score: s.physics_score,
            cosine_score: s.cosine_score,
            physics_score: s.physics_score,
            hamming_score: null,
            keyword_boost: null,
            title: title.trim(),
            preview: passage.slice(0, 240),
            full_text: passage,
            from_lattice: true,
            prev_idx: null,
            next_idx: null,
        } as SearchResult;
    });
}

/**
 * Convenience: true if a cart's brain is already cached in the engine.
 * Lets the UI hide "Loading brain…" when the answer will be instant.
 */
export function isBrainLoadedFor(cartName: string): boolean {
    return brainLoaded.has(cartName);
}
