"""Pydantic request/response models for Vector+ Studio API."""

from pydantic import BaseModel, Field


# --- Requests ---

class MountRequest(BaseModel):
    filename: str

class SearchRequest(BaseModel):
    query: str
    mode: str = "smart"       # "smart" | "pure_brain" | "fast"
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
    title: str
    preview: str
    full_text: str
    from_lattice: bool = False

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

class DeletedPattern(BaseModel):
    idx: int
    title: str
    preview: str

class DeletedListResponse(BaseModel):
    deleted: list[DeletedPattern]

class MessageResponse(BaseModel):
    success: bool
    message: str
