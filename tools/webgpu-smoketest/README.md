# WebGPU Cart Builder — smoke-test artifacts + Thursday TODO

Filed: 2026-05-06 evening. Smoke tests retired the three big risks before
betting Thursday-Friday on the WebGPU pivot.

## Smoke-test results (2026-05-06, ~9pm)

### Test 1 — WebGPU lattice engine alive

Existing engine at [vector-benchmark-demo/cuda/webgpu/lattice-engine.js](../../../vector-benchmark-demo/cuda/webgpu/lattice-engine.js) verified working in Chrome on Windows:

```
WebGPU available
Initializing 4096x4096 engine...
Engine ready: 4096x4096
Random imprint: 151.7ms
Train 30f: 317.8ms, 1,290,188 active (7.7%)
Signature generated: 48.8ms, 4096 regions
L4 state recalled
Viewing L3 (256x256)
```

Implications: 12ms/frame Hebbian on 16.7M neurons, 49ms signature gen — a
200-passage cart's lattice work is order-of-seconds, not minutes.

### Test 2 — Embedder parity (Python vs in-browser transformers.js)

Same sentence (`"search_document: this is a test sentence about transformer attention"`)
embedded both ways. Output drift at the 6th decimal place — noise floor.

```
Python:  L2_NORM=21.072100  FIRST_10=[-0.568762, 0.602091, -3.335682, ...]
Browser: L2_NORM=21.072095  FIRST_10=[-0.568761, 0.602090, -3.335682, ...]
```

Carts built in browser will be bit-near-identical to carts built server-side.
Cross-compatibility confirmed.

Reference vector saved to [python_reference_full.npy](python_reference_full.npy)
for future cosine-similarity comparison if Nomic ever ships a new ONNX variant.

### Test 3 — NPZ writer feasibility (assessed via membraine)

`npyjs` (the library I incorrectly assumed was read-only earlier) has a
maintained `dump()` writer:

```javascript
import { dump } from "npyjs";
const bytes = dump(new Float32Array([1, 2, 3, 4]), [2, 2]);
```

Supports every dtype our cart format uses (int8/uint8/int16/uint16/int32/uint32/
int64/uint64/float32/float64/float16/complex64/complex128). Apache-2.0,
maintained, modern Node 18+/browsers/Deno/Bun.

NPZ container = `npyjs.dump()` outputs zipped via JSZip. ~30-50 lines total
for the cart writer.

## Thursday TODO — Cart Builder V2 (WebGPU)

### Block 1 — Document parsers (~3-4h)

Goal: drag a PDF/DOCX/XLSX in, get parsed text + section metadata out.

- [ ] `frontend/src/cart-builder-v2/parsers/pdf.ts` — wrap `pdfjs-dist`
- [ ] `frontend/src/cart-builder-v2/parsers/docx.ts` — wrap `mammoth`
- [ ] `frontend/src/cart-builder-v2/parsers/xlsx.ts` — wrap `xlsx` (SheetJS)
- [ ] `frontend/src/cart-builder-v2/parsers/text.ts` — txt/md/rtf trivial cases
- [ ] `frontend/src/cart-builder-v2/parsers/index.ts` — dispatch by file extension/MIME, return `{sections: Section[], metadata: ParseMeta}`
- [ ] `frontend/src/cart-builder-v2/chunker.ts` — port the chunking logic from `cart-builder/cart-builder/parsers.py`

Reference: existing Flask `parsers.py` is the canonical chunking behavior.
Match its output shape so pre-existing carts stay identifiable.

### Block 2 — Embedder integration (~2-3h)

Goal: chunks → 768-dim Float32Array embeddings, all in-browser.

- [ ] `frontend/src/cart-builder-v2/embedder/loader.ts` — load Nomic ONNX once, cache instance
- [ ] `frontend/src/cart-builder-v2/embedder/embed.ts` — batch embed N chunks, return `Float32Array` of shape `[N, 768]`
- [ ] Progress callback API for UI (model download + per-chunk embed)
- [ ] Error handling: fall back from WebGPU to WASM if model load fails on the GPU path

Use exactly the same prefix convention as Python: `search_document: <chunk text>`.
Verified parity in Test 2.

### Block 3 — Cart writer (~1-2h, was estimated half-day, npyjs reduced it)

Goal: in-memory cart structure → downloadable `.cart.npz` + sidecars.

- [ ] `frontend/src/cart-builder-v2/writer/npy.ts` — wrap `npyjs.dump`
- [ ] `frontend/src/cart-builder-v2/writer/hippocampus.ts` — pack the 64-byte struct (offsets confirmed against [api/cartridge_io.py](../../api/cartridge_io.py): `<I B B I I I I H I B B 34s`). Remember: `flags` byte at offset 28 = membot's tombstone/pinned/has_parent/etc; `perms_byte` at offset 29 = our Step 2b RWX
- [ ] `frontend/src/cart-builder-v2/writer/manifest.ts` — JSON manifest with fingerprint + version + count
- [ ] `frontend/src/cart-builder-v2/writer/permissions.ts` — `.permissions.json` sidecar (default `r` for browser-built)
- [ ] `frontend/src/cart-builder-v2/writer/npz.ts` — bundle embeddings.npy + passages + hippocampus + signatures via JSZip
- [ ] Trigger download via `<a href="blob:..." download={cart_name + '.cart.npz'}>`

### Block 4 — Pipeline + UI integration (~2-3h)

Goal: wire it into the existing CartBuilderScreen.

- [ ] `frontend/src/cart-builder-v2/pipeline.ts` — orchestrator: parse → chunk → embed → write → download
- [ ] Progress events for UI (file parsed, chunk embedded, cart written)
- [ ] Add `enableWebGPUBuilder` feature flag → render new UI when on, fall back to existing server-side flow when off
- [ ] CartBuilderScreen swap: in read-only mode + WebGPU available, show the V2 pipeline; otherwise show the existing flow
- [ ] Build progress UI (per-file + per-chunk + overall)

### Block 5 — Parity test (~1h, the deal-breaker confirm)

Goal: same inputs through server-side vs browser-side build → same cart.

- [ ] Take the existing `attention-is-all-you-need.cart.npz` reference
- [ ] Build same source PDF via browser pipeline
- [ ] Mount both, run same query, verify result rankings agree

If parity fails: investigate. Likely culprits in priority order:
1. Different chunking (most likely)
2. Different prefix handling
3. Hippocampus row layout mismatch
4. Floating-point drift in embeddings (Test 2 says this is fine)

## Friday TODO — Polish + ship

- Bundle audit (the WebGPU model is ~80MB; first-load progress UI is mandatory)
- Fall-back path when WebGPU unavailable (transformers.js falls back to WASM automatically; verify it still works end-to-end)
- Document size cap + chunk count cap (to prevent low-end machines from chewing through 1GB of RAM)
- "Cart Builder V2" in-app banner explaining the new pipeline
- Marketing copy: "your data never leaves your machine"
- Optional stretch: physics-trained cart variant via the WebGPU lattice engine

## Pre-installed deps (committed Wed evening)

```
"@huggingface/transformers": "^4.2.0",  -- in-browser ONNX runtime
"npyjs":                     "^1.1.0",  -- NPY read + write
"jszip":                     "^3.10.1", -- NPZ container
"pdfjs-dist":                "^5.6.205", -- PDF parser
"mammoth":                   "^1.12.0", -- DOCX parser
"xlsx":                      "^0.18.5"  -- XLSX (SheetJS)
```

## Smoke-test artifacts

- [embedder-test.html](embedder-test.html) — re-run any time to verify embedder parity
- [python_reference.py](python_reference.py) — Python reference embedding generator
- [python_reference_full.npy](python_reference_full.npy) — full 768-dim reference vector

To re-run end-to-end: spin up an HTTP server in this directory (`python -m http.server 9876`), open `http://127.0.0.1:9876/embedder-test.html`, click Run test, paste output back.
