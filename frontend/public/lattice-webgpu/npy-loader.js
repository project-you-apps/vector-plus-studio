/**
 * Minimal NumPy .npy v1/v2/v3 parser for the brain-fetch path.
 *
 * Format reference: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
 *   bytes 0..5   : magic '\x93NUMPY'
 *   bytes 6..7   : version (major, minor)
 *   bytes 8..    : header length (uint16 LE for v1, uint32 LE for v2/v3)
 *   then         : ASCII/UTF-8 header dict ending with newline
 *   then         : raw data
 *
 * We only need the legacy brain format: dtype='<u4', 1-D shape=(N,).
 * Returns the raw view + dtype + shape so the caller can decide what to do.
 */

const NPY_MAGIC = '\x93NUMPY';

/**
 * Parse a .npy file fetched as ArrayBuffer.
 * @param {ArrayBuffer} buffer
 * @returns {{ dtype: string, shape: number[], fortranOrder: boolean, data: ArrayBuffer, dataOffset: number }}
 */
export function parseNpy(buffer) {
    const bytes = new Uint8Array(buffer);

    let magic = '';
    for (let i = 0; i < 6; i++) magic += String.fromCharCode(bytes[i]);
    if (magic !== NPY_MAGIC) {
        throw new Error(`Not a .npy file: expected magic '${NPY_MAGIC}', got '${magic}'`);
    }

    const major = bytes[6];
    const minor = bytes[7];

    let headerLen;
    let dataOffset;
    if (major === 1) {
        const view = new DataView(buffer, 8, 2);
        headerLen = view.getUint16(0, true);
        dataOffset = 10 + headerLen;
    } else if (major === 2 || major === 3) {
        const view = new DataView(buffer, 8, 4);
        headerLen = view.getUint32(0, true);
        dataOffset = 12 + headerLen;
    } else {
        throw new Error(`Unsupported .npy version ${major}.${minor}`);
    }

    let header = '';
    const headerStart = major === 1 ? 10 : 12;
    for (let i = headerStart; i < headerStart + headerLen; i++) {
        header += String.fromCharCode(bytes[i]);
    }

    const dtypeMatch = header.match(/'descr'\s*:\s*'([^']+)'/);
    const fortranMatch = header.match(/'fortran_order'\s*:\s*(True|False)/);
    const shapeMatch = header.match(/'shape'\s*:\s*\(([^)]*)\)/);

    if (!dtypeMatch || !shapeMatch) {
        throw new Error(`Could not parse .npy header: ${header}`);
    }

    const dtype = dtypeMatch[1];
    const fortranOrder = fortranMatch ? fortranMatch[1] === 'True' : false;
    const shape = shapeMatch[1]
        .split(',')
        .map(s => s.trim())
        .filter(s => s.length > 0)
        .map(s => parseInt(s, 10));

    return {
        dtype,
        shape,
        fortranOrder,
        data: buffer.slice(dataOffset),
        dataOffset,
    };
}

/**
 * Fetch a .npy file with progress callback.
 * @param {string} url
 * @param {(loaded: number, total: number) => void} [onProgress]
 * @returns {Promise<ArrayBuffer>}
 */
export async function fetchNpyWithProgress(url, onProgress) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Fetch ${url} failed: ${response.status} ${response.statusText}`);
    }

    const totalStr = response.headers.get('content-length');
    const total = totalStr ? parseInt(totalStr, 10) : 0;

    if (!onProgress || !response.body) {
        return response.arrayBuffer();
    }

    const reader = response.body.getReader();
    const chunks = [];
    let loaded = 0;
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.length;
        onProgress(loaded, total);
    }

    const buffer = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
        buffer.set(chunk, offset);
        offset += chunk.length;
    }
    return buffer.buffer;
}
