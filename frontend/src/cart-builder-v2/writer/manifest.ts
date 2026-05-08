// Manifest format must match membot/membot_server.py:save_manifest exactly,
// or mount will refuse the cart with FINGERPRINT MISMATCH.
//
// Reference: tools/bootstrap_claude_journal.py:compute_membot_fingerprint
//   sha256(embeddings[0].tobytes() + embeddings[-1].tobytes() + str(count).encode())[:16]
// When count == 1, last = first (offsets coincide). We mirror that.

export const CART_MANIFEST_VERSION = 'mcp-v3'

export interface CartManifest {
  version: string
  count: number
  fingerprint: string
  timestamp: string
}

export async function buildManifest(
  embeddings: Float32Array,
  count: number,
  dim: number
): Promise<CartManifest> {
  return {
    version: CART_MANIFEST_VERSION,
    count,
    fingerprint: await computeFingerprint(embeddings, count, dim),
    timestamp: new Date().toISOString(),
  }
}

export async function computeFingerprint(
  embeddings: Float32Array,
  count: number,
  dim: number
): Promise<string> {
  if (count === 0) {
    throw new Error('Cannot compute fingerprint for zero embeddings')
  }
  if (embeddings.length !== count * dim) {
    throw new Error(
      `Fingerprint: embeddings length ${embeddings.length} ≠ count*dim ${count * dim}`
    )
  }

  const bytesPerVec = dim * 4 // float32 = 4 bytes per element
  const firstBytes = new Uint8Array(
    embeddings.buffer,
    embeddings.byteOffset,
    bytesPerVec
  )
  // Last vector: when count==1 this aliases firstBytes — Python does the
  // same (last = embeddings[0] when count==1).
  const lastBytes = new Uint8Array(
    embeddings.buffer,
    embeddings.byteOffset + (count - 1) * bytesPerVec,
    bytesPerVec
  )
  const countBytes = new TextEncoder().encode(String(count))

  const combined = new Uint8Array(bytesPerVec * 2 + countBytes.length)
  combined.set(firstBytes, 0)
  combined.set(lastBytes, bytesPerVec)
  combined.set(countBytes, bytesPerVec * 2)

  const hashBuffer = await crypto.subtle.digest('SHA-256', combined)
  const hashArray = new Uint8Array(hashBuffer)
  const hex = Array.from(hashArray, (b) => b.toString(16).padStart(2, '0')).join('')
  return hex.slice(0, 16)
}
