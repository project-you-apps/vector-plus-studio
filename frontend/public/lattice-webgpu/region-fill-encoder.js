/**
 * Region-Fill Encoder — JS twin of api/region_fill_encoder.py + the
 * _ab_encode_region_fill / _ab_decode_region_fill_analog helpers in api/search.py
 * used by the two-sided Associate v83 algorithm.
 *
 * The 4D→2D reshape (64,64,64,64) → (4096,4096) in C-order means each
 * embedding dim's "region" actually occupies one full 4096-wide row in 2D,
 * not a 64×64 block.
 * For dim i (i in 0..n_dims-1), 2D row = i, all 4096 cols.
 */

const LATTICE_SIZE = 4096;
const DEFAULT_N_DIMS = 768;

/**
 * Encode a 768-dim float embedding to a 4096×4096 binary pattern.
 * @param {Float32Array | number[]} embedding - 768-dim vector
 * @param {number} nDims
 * @returns {Uint8Array} flat 16,777,216-byte pattern (0 or 1)
 */
export function abEncodeRegionFill(embedding, nDims = DEFAULT_N_DIMS) {
    const pattern = new Uint8Array(LATTICE_SIZE * LATTICE_SIZE);
    for (let i = 0; i < nDims; i++) {
        if (embedding[i] > 0) {
            const rowStart = i * LATTICE_SIZE;
            pattern.fill(1, rowStart, rowStart + LATTICE_SIZE);
        }
    }
    return pattern;
}

/**
 * Analog decode of a settled lattice to a per-dim region-mean vector in [0, 1].
 * No threshold — preserves continuous activation signal matching the v83
 * thermometer decode. Binary thresholding collapses sparse settled states
 * (cosine of all-zero == 0).
 *
 * @param {Float32Array | Uint8Array} lattice - flat 16,777,216 cells (or 4096×4096)
 * @param {number} nDims
 * @returns {Float32Array} nDims-length analog vector in [0, 1]
 */
export function abDecodeRegionFillAnalog(lattice, nDims = DEFAULT_N_DIMS) {
    const out = new Float32Array(nDims);
    for (let i = 0; i < nDims; i++) {
        const rowStart = i * LATTICE_SIZE;
        let sum = 0;
        for (let j = 0; j < LATTICE_SIZE; j++) {
            sum += lattice[rowStart + j];
        }
        out[i] = sum / LATTICE_SIZE;
    }
    return out;
}

/**
 * L2-normalize a vector in place and return it.
 * @param {Float32Array} v
 * @returns {Float32Array}
 */
export function l2Normalize(v) {
    let sumSq = 0;
    for (let i = 0; i < v.length; i++) sumSq += v[i] * v[i];
    const norm = Math.sqrt(sumSq) + 1e-9;
    for (let i = 0; i < v.length; i++) v[i] /= norm;
    return v;
}

/**
 * Dot product of two equal-length vectors.
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @returns {number}
 */
export function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
}
