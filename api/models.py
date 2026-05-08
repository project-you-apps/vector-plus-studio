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
    cart_permissions: dict | None = None  # cart-format RWX sidecar (Step 2a)
    # True when the currently-mounted cart's file lives inside the upload
    # sandbox (cartridges/_session_uploads/). UI uses this to surface an
    # "Eject" button that immediately purges the user's uploaded cart.
    mounted_is_sandboxed: bool = False
    # Absolute path of the mounted cart, or None. Frontend passes this back
    # to /api/cartridges/eject to identify which sandbox file to delete.
    mounted_path: str | None = None

class DeletedPattern(BaseModel):
    idx: int
    title: str
    preview: str

class DeletedListResponse(BaseModel):
    deleted: list[DeletedPattern]

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

class MessageResponse(BaseModel):
    success: bool
    message: str


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
