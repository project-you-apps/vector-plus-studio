// Thin wrapper around npyjs.dump.
//
// npyjs supports numeric dtypes AND unicode string arrays (<U<n>).
// Browser-built carts use unicode NPY for passages, so the on-disk format
// stays byte-near-identical to Python-built carts (np.array(strs, dtype='<U<n>')).
// No server-side mount-path change required.

import { dump } from 'npyjs'

export function dumpFloat32(data: Float32Array, shape: number[]): ArrayBuffer {
  return dump(data, shape)
}

export function dumpUint8(data: Uint8Array, shape: number[]): ArrayBuffer {
  return dump(data, shape)
}

export function dumpInt32(data: Int32Array, shape: number[]): ArrayBuffer {
  return dump(data, shape)
}

/**
 * Write a string array as a numpy unicode NPY (`<U<n>` dtype, where N is the
 * max string length). numpy loads this directly with `np.load(...)` without
 * `allow_pickle=True`. Cleaner than object-array-pickle for cross-language
 * compat.
 */
export function dumpUnicode(data: string[], shape: number[]): ArrayBuffer {
  return dump(data, shape)
}
