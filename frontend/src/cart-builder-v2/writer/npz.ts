import JSZip from 'jszip'
import type { Section } from '../types'
import { NOMIC_DIM } from '../embedder/embed'
import { type HippocampusOptions, packHippocampus } from './hippocampus'
import { type CartManifest, buildManifest } from './manifest'
import { dumpFloat32, dumpUint8, dumpUnicode } from './npy'
import {
  type CartPermissionsPayload,
  type CartPermissionsSpec,
  buildPermissions,
} from './permissions'

export interface BuildCartOptions {
  cartName: string
  hippocampus?: HippocampusOptions
  permissions?: CartPermissionsSpec
}

export interface BuiltCart {
  /** ZIP container with embeddings.npy + hippocampus.npy + passages.json + compressed_texts.json. */
  cartBlob: Blob
  /** Standalone .cart_manifest.json (mcp-v3 fingerprint). */
  manifestBlob: Blob
  /** Standalone .permissions.json (Step 2a). */
  permissionsBlob: Blob
  manifest: CartManifest
  permissions: CartPermissionsPayload
  cartFilename: string
  manifestFilename: string
  permissionsFilename: string
}

/**
 * Bundle embeddings + sections into a deployable cart trio.
 *
 * The .cart.npz is a ZIP container holding:
 *   embeddings.npy        — Float32Array shape [N, 768], NPY float32
 *   hippocampus.npy       — Uint8Array shape [N, 64], NPY uint8
 *   passages.npy          — string[] shape [N], NPY unicode (<U<maxlen>)
 *   compressed_texts.npy  — string[] shape [N], NPY unicode (no compression in v1)
 *
 * passages and compressed_texts use NPY unicode (`<U<n>`) rather than the
 * pickled object-array Python defaults to. numpy loads `<U` arrays without
 * `allow_pickle=True` and behaves identically for read purposes (trailing
 * nulls stripped on access). Net result: browser-built carts mount on the
 * existing membot server with no server-side changes needed.
 */
export async function buildCart(
  embeddings: Float32Array,
  sections: Section[],
  options: BuildCartOptions
): Promise<BuiltCart> {
  const count = sections.length
  if (count === 0) {
    throw new Error('Cannot build cart from zero sections')
  }
  if (embeddings.length !== count * NOMIC_DIM) {
    throw new Error(
      `Embeddings size mismatch: ${embeddings.length} (expected ${count * NOMIC_DIM} for ${count} sections × ${NOMIC_DIM} dims)`
    )
  }

  const passages = sections.map((s) => s.text)
  const compressed_texts = passages // v1: no compression
  const hippocampus = packHippocampus(sections, options.hippocampus)
  const manifest = await buildManifest(embeddings, count, NOMIC_DIM)
  const permissions = buildPermissions(options.permissions)

  const zip = new JSZip()
  zip.file('embeddings.npy', dumpFloat32(embeddings, [count, NOMIC_DIM]))
  zip.file('hippocampus.npy', dumpUint8(hippocampus, [count, 64]))
  zip.file('passages.npy', dumpUnicode(passages, [count]))
  zip.file('compressed_texts.npy', dumpUnicode(compressed_texts, [count]))

  const cartBlob = await zip.generateAsync({
    type: 'blob',
    compression: 'DEFLATE',
    compressionOptions: { level: 6 },
  })
  const manifestBlob = new Blob([JSON.stringify(manifest, null, 2)], {
    type: 'application/json',
  })
  const permissionsBlob = new Blob([JSON.stringify(permissions, null, 2)], {
    type: 'application/json',
  })

  return {
    cartBlob,
    manifestBlob,
    permissionsBlob,
    manifest,
    permissions,
    cartFilename: `${options.cartName}.cart.npz`,
    manifestFilename: `${options.cartName}.cart_manifest.json`,
    permissionsFilename: `${options.cartName}.permissions.json`,
  }
}

/**
 * Save the three cart artifacts. Uses the File System Access API
 * (`showDirectoryPicker`) when available so the user can pick a real
 * destination folder instead of being forced into the browser's
 * Downloads/ default — Chrome/Edge/Opera 86+ support this. Firefox and
 * Safari fall back to the legacy three-<a download> approach.
 *
 * Resolves on completion (or user cancel). Rejects only on unexpected
 * filesystem-API failure; caller should still treat a rejection as
 * non-fatal since the user can always re-trigger the save.
 */
export async function downloadBuiltCart(cart: BuiltCart): Promise<void> {
  type DirPickerOpts = { mode?: 'read' | 'readwrite' }
  type DirPicker = (opts?: DirPickerOpts) => Promise<FileSystemDirectoryHandle>
  const pickerFn = (window as unknown as { showDirectoryPicker?: DirPicker }).showDirectoryPicker
  if (typeof pickerFn === 'function') {
    try {
      const dir = await pickerFn({ mode: 'readwrite' })
      await writeBlobToDir(dir, cart.cartFilename, cart.cartBlob)
      await writeBlobToDir(dir, cart.manifestFilename, cart.manifestBlob)
      await writeBlobToDir(dir, cart.permissionsFilename, cart.permissionsBlob)
      return
    } catch (err) {
      // User cancelled the picker — bail silently, no fallback (legacy
      // download would land files they didn't ask for).
      if ((err as DOMException)?.name === 'AbortError') return
      // Permission denied / quota / other API failure — fall through to
      // the legacy path so the user still gets the cart somewhere.
      console.warn('showDirectoryPicker failed, falling back to <a download>', err)
    }
  }
  triggerDownload(cart.cartBlob, cart.cartFilename)
  triggerDownload(cart.manifestBlob, cart.manifestFilename)
  triggerDownload(cart.permissionsBlob, cart.permissionsFilename)
}

async function writeBlobToDir(
  dir: FileSystemDirectoryHandle,
  name: string,
  blob: Blob,
): Promise<void> {
  const fileHandle = await dir.getFileHandle(name, { create: true })
  const writable = await fileHandle.createWritable()
  await writable.write(blob)
  await writable.close()
}

function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.style.display = 'none'
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  // Revoke after a tick so the browser has time to start the download.
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
