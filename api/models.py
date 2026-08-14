"""Pydantic request/response models for Vector+ Studio API."""

from pydantic import BaseModel, Field


# --- Requests ---

class MountRequest(BaseModel):
    filename: str

class SearchRequest(BaseModel):
    query: str
    mode: str = "hamming"     # "hamming" | "smart" | "pure_brain" | "fast" | "associate"
    alpha: float = 0.7        # blend slider (0.0-1.0), only for "smart"
    top_k: int = 10

class ForgeRequest(BaseModel):
    name: str

class AddPassageRequest(BaseModel):
    text: str

class RestoreRequest(BaseModel):
    idx: int


# --- Responses ---

class CartridgeInfo(BaseModel):
    name: str
    filename: str
    size_mb: float
    has_brain: bool
    has_signatures: bool
    has_manifest: bool
    pattern_count: int | None = None

class CartridgeListResponse(BaseModel):
    cartridges: list[CartridgeInfo]

class MountResponse(BaseModel):
    success: bool
    name: str = ""
    pattern_count: int = 0
    multimodal: bool = False
    brain_loaded: bool = False
    signatures_loaded: bool = False
    message: str = ""
    # 2026-07-23 -- Permission UI MVP #2. Cart's .permissions.json sidecar
    # payload, or None if the cart has no sidecar (legacy carts default to
    # writable). Frontend reads this to know whether Edit Carts should
    # surface the read-only banner + disable destructive actions.
    # Populated by _apply_cart_permissions_after_mount() which loads the
    # sidecar and sets engine.cart_permissions. Shape mirrors
    # api/cartridge_io.py:load_cart_permissions -- typically
    # {"default": "r" | "rw", "version": "1.0", "owner": "...",
    #  "description": "..."} but any subset is valid.
    cart_permissions: dict | None = None

class SearchResult(BaseModel):
    rank: int
    idx: int
    score: float
    cosine_score: float | None = None
    physics_score: float | None = None
    hamming_score: float | None = None
    keyword_boost: float | None = None
    title: str
    preview: str
    full_text: str
    from_lattice: bool = False
    prev_idx: int | None = None
    next_idx: int | None = None
    # Split-cart provenance hints — populated when the mounted cart has a
    # SQLite sidecar. Frontend uses presence of source_db to render the
    # "Load full passage from <db>" CTA in the modal (parity with membot's
    # RAG+ provenance feature). paper_id arrives after the source-load fetch.
    source_db: str | None = None
    paper_id: str | None = None
    # Step 2b: per-pattern RWX from the hippocampus row's flags byte.
    perms: dict | None = None
    # v3 provenance — per-pattern source filename. Populated by the search
    # endpoint from the mounted cart's source_paths cache (which reads the
    # v3 source_strings table + h-row source_idx natively, falling back to
    # the v1 source_paths.npy sidecar for older carts). ResultCard renders
    # this as a "from <filename>" caption above the title when populated.
    # None when the cart carries no provenance surface at all.
    source_path: str | None = None
    # v3 provenance — h-row ingestion timestamp (bytes 24-27 uint32 LE)
    # formatted as ISO 8601 UTC. Same field as PatternResponse.ingested_at
    # so PassageModal can render "Ingested" consistently whether opened from
    # a fresh search or navigated to via Prev/Next.
    ingested_at: str | None = None

class SearchResponse(BaseModel):
    query: str
    mode: str
    elapsed_ms: float
    result_count: int
    results: list[SearchResult]

class StatusResponse(BaseModel):
    engine_ready: bool
    gpu_available: bool
    mounted_cartridge: str | None = None
    pattern_count: int = 0
    physics_trained: bool = False
    training_active: bool = False
    training_progress: int = 0
    training_total: int = 0
    multimodal: bool = False
    signatures_loaded: bool = False
    deleted_count: int = 0
    dirty: bool = False
    read_only: bool = True
    read_only_mode: bool = False  # global server-side lock (VPS_READ_ONLY env var)
    # True when this server may not expose arbitrary filesystem paths.
    # Distinct from read_only_mode: one governs WRITES, this governs whether a
    # stranger can walk the disk. Split 2026-08-06 -- see main.PUBLIC_HOST.
    public_host: bool = False
    cart_permissions: dict | None = None  # cart-format RWX sidecar (Step 2a)
    # True when the currently-mounted cart's file lives inside the upload
    # sandbox (cartridges/_session_uploads/). UI uses this to surface an
    # "Eject" button that immediately purges the user's uploaded cart.
    mounted_is_sandboxed: bool = False
    # WHO ELSE is looking at the mounted cart, by display name, excluding the caller.
    # Empty when nobody else is, when the cart is withheld, or when the caller may not read
    # it -- occupancy is personnel information and rides the same gate as the cart's NAME.
    # Display names only: a seat uuid is noise to a person and a disclosure to everyone else.
    cart_occupants: list[str] = []
    # NO mounted_path. It was here so the frontend could pass the absolute path back to
    # /api/cartridges/eject, which meant /api/status -- an endpoint the app polls every 2s
    # BEFORE sign-in, so it cannot be behind auth -- handed the full server path to anyone
    # who asked. Removed 2026-08-10 along with eject's path parameter: the server already
    # knows which cart is mounted and does not need the client to tell it what to delete.
    # 2026-08-03 — per-seat attention. Reported even when OFF, and with the reason:
    # a silently-absent feature is indistinguishable from a broken one, and this one
    # can be off for three different reasons (no modules, no seat, no cart).
    seat_attention: dict | None = None

# --- Pattern-0 TOC (v1) ---
# 2026-07-01 spec: left-side TOC panel on the Search tab that shows metadata
# + table-of-contents of the currently-mounted cart. Read-only v1.
# Backend endpoint: GET /api/cart/pattern-0

class Pattern0TocItem(BaseModel):
    name: str
    description: str | None = None
    pattern_count: int = 0


class Pattern0Response(BaseModel):
    mounted: bool = False
    # Full Pattern-0 metadata (v1 header). When the mounted cart lacks a
    # meaningful Pattern-0, `is_derived=True` and these come from source_paths
    # + engine.mounted_name fallback.
    name: str | None = None
    description: str | None = None
    creator: str | None = None
    created_at: str | None = None  # ISO 8601
    owner: str | None = None
    pattern_count: int = 0
    # v2 agent_briefing block. Only present when the cart's Pattern-0 has one.
    # See docs/RFC/pattern-0-v2-spec.md.
    agent_briefing: str | None = None
    # Flat list of files (or free-floating passages) in the cart. Client-side
    # substring filter narrows the visible list without a round-trip.
    toc_items: list[Pattern0TocItem] = Field(default_factory=list)
    # True when Pattern-0 was minimal/absent and toc_items were computed from
    # unique source_paths in the hippocampus rows (or from the raw passages
    # array for legacy carts).
    is_derived: bool = False


class PerPatternMetaResponse(BaseModel):
    """Per-pattern metadata sidecar (from per_pattern_meta.npy).

    exists so sandbox-mounted carts on the droplet
    reach parity with LocalCart-mounted carts for image/table rendering.
    The frontend fetches this on mount + carries a mirror-image data
    structure through ResultCard, PassageModal, Pattern0TocPanel drill-down,
    and Edit Carts source-files panel.

    Records parallel `passages` — records[i] is the metadata for pattern i.
    Each record's shape mirrors the JSON baked by the writer (see
    api/cartbuilder/builder.py + frontend/src/cart-builder-v2/writer/npz.ts):
      - v: schema version (1)
      - content_type: "document" | "graphic" | "table"
      - source: original filename
      - page: 1-indexed page or null
      - chunk / chunks: position within section
      - tags: list[str]
      - created_at: unix ts
      - tombstone: bool
      - caption, image_b64, bbox (graphic)
      - html, bbox (table)
    """
    mounted: bool = False
    records: list[dict] = Field(default_factory=list)


class DeletedPattern(BaseModel):
    idx: int
    title: str
    preview: str

class DeletedListResponse(BaseModel):
    deleted: list[DeletedPattern]

class PatternListItem(BaseModel):
    idx: int
    title: str
    preview: str
    word_count: int

class PatternListResponse(BaseModel):
    passages: list[PatternListItem]
    total: int          # active (non-tombstoned) passage count, post-filter
    offset: int
    limit: int
    filter: str | None = None

class PatternResponse(BaseModel):
    idx: int
    title: str
    preview: str
    full_text: str
    prev_idx: int | None = None
    next_idx: int | None = None
    # Split-cart provenance — populated when /api/patterns/{idx} fetched the
    # full text from a SQLite sidecar (parity with membot RAG+ provenance).
    source_db: str | None = None
    paper_id: str | None = None
    # Step 2b: per-pattern RWX from the hippocampus row's flags byte. None
    # if the cart has no hippocampus or this pattern's flags=0 (legacy).
    perms: dict | None = None
    # v3 provenance — per-pattern source filename, resolved from
    # source_strings + h-row source_idx (or v1 sidecar fallback). Populated
    # for local + hosted carts alike; None when the cart carries no
    # provenance data at all. Same field as SearchResult.source_path so the
    # PassageModal can render a consistent "Source: filename" line.
    source_path: str | None = None
    # v3 provenance — h-row ingestion timestamp (bytes 24-27 uint32 LE)
    # formatted as ISO 8601 UTC. Populated at cart-build time; batch-scoped
    # for CLI-built carts (every pattern in one build shares the batch
    # timestamp) and per-exchange for session-cart auto-append. None when
    # no hippocampus present or timestamp is zero.
    ingested_at: str | None = None

class MessageResponse(BaseModel):
    success: bool
    message: str
    # Set by PUT /api/patterns/{idx}. An edit can SUCCEED while its succession link fails to
    # record — the passage really was replaced, so returning success=False would be a lie —
    # but that leaves clusters and per-seat attention pointing at text nobody will find
    # again. A structured flag so a client can render a real warning instead of
    # string-matching the message prose. None means "not an edit; not applicable".
    succession_recorded: bool | None = None


# --- Membox visualizer ---

class MemboxLockState(BaseModel):
    cart_id: str
    holder: str | None = None
    held_for_seconds: float | None = None
    lease_seconds: int = 30
    acquire_count: int = 0
    wait_count: int = 0
    is_locked: bool = False

class MemboxCartInfo(BaseModel):
    cart_id: str
    role: str | None = None
    n_patterns: int = 0
    lock: MemboxLockState
    recent_writes: int = 0

class MemboxWriteEntry(BaseModel):
    agent_id: str
    written_at: str
    local_addr: int
    origin: str
    text_preview: str

class MemboxStatus(BaseModel):
    cart_id: str
    n_patterns: int
    lock: MemboxLockState
    writes_by_agent: dict[str, int] = Field(default_factory=dict)
    recent_writes: list[MemboxWriteEntry] = Field(default_factory=list)
    membox_enabled: bool = True

class MemboxCartListResponse(BaseModel):
    carts: list[MemboxCartInfo]

class MemboxImprintRequest(BaseModel):
    cart_id: str
    text: str
    agent_id: str
    tags: str = ""
    reasoning: str = ""
    origin: str = "agent"
    timeout_ms: int = 5000

class MemboxMountRequest(BaseModel):
    cart_path: str
    cart_id: str | None = None
    role: str | None = None
    lease_seconds: int = 30
    verify_integrity: bool = True

class MemboxUnmountRequest(BaseModel):
    cart_id: str
