// Cart Builder V2 — in-browser pipeline (parsers → chunker → embedder → writer).
// Block 1 (parsers + chunker) shipped 2026-05-08.
// Block 2 (embedder) shipped 2026-05-08.
// Block 3 (writer) shipped 2026-05-08.
// Pipeline orchestrator + UI integration land in Block 4.

export type {
  ChunkOptions,
} from './chunker'
export { chunkSections } from './chunker'

export {
  listParsers,
  parseFile,
  registerParser,
} from './parsers'

export type {
  EmbedderBackend,
  ModelDownloadProgress,
  ParseError,
  ParseMeta,
  ParseResult,
  Parser,
  ProgressCallback,
  Section,
} from './types'

export type { LoaderOptions } from './embedder/loader'
export {
  clearEmbedderCache,
  getActiveBackend,
  getEmbedder,
} from './embedder/loader'

export type { EmbedOptions, EmbedResult, PrefixMode } from './embedder/embed'
export { embedQuery, embedTexts, NOMIC_DIM } from './embedder/embed'

// Writer (Block 3)
export type { HippocampusOptions } from './writer/hippocampus'
export {
  HIPPO_SIZE,
  PERM_BROWSER_DEFAULT,
  PERM_R,
  PERM_W,
  PERM_X,
  packHippocampus,
} from './writer/hippocampus'

export type { CartManifest } from './writer/manifest'
export {
  CART_MANIFEST_VERSION,
  buildManifest,
  computeFingerprint,
} from './writer/manifest'

export type {
  CartPermissionsPayload,
  CartPermissionsSpec,
  DefaultPermsString,
} from './writer/permissions'
export {
  PERMISSIONS_SCHEMA_VERSION,
  buildPermissions,
} from './writer/permissions'

export type { BuildCartOptions, BuiltCart } from './writer/npz'
export { buildCart, downloadBuiltCart } from './writer/npz'

// Pipeline (Block 4)
export type {
  PipelineOptions,
  PipelineProgress,
  PipelineResult,
  PipelineStage,
} from './pipeline'
export {
  buildCartFromFiles,
  buildCartFromPassages,
  DEFAULT_MAX_CHUNKS_PER_BUILD,
  DEFAULT_MAX_FILE_SIZE_BYTES,
} from './pipeline'
