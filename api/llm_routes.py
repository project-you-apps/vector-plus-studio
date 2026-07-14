"""FastAPI route wiring for the LLM adapter (Track C).

Exposes ``POST /api/llm/synthesize`` on top of the provider-agnostic
``LLMAdapter`` interface (``api/llm/adapter.py``). The concrete provider
(Cloudflare Workers AI, Anthropic API, Heartbeat BYO Claude) is selected
at runtime by ``get_llm_adapter()`` based on the
``VECTOR_PLUS_LLM_PROVIDER`` environment variable.

Wave 2 reports (Timeline, Trend, Financial Rollup, Executive TL;DR) call
this endpoint from their generate() implementations for LLM-assisted
extraction and synthesis. Wave 1 reports do not touch it.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .llm.adapter import LLMError
from .llm.registry import get_llm_adapter


router = APIRouter(prefix="/api/llm", tags=["llm"])


class SynthesizeRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model_hint: str = "default"
    max_tokens: int = Field(2048, ge=1, le=16384)


class SynthesizeResponse(BaseModel):
    text: str
    provider: str
    model: str
    tokens_used: Optional[int] = None
    neurons_used: Optional[int] = None
    cost_usd: Optional[float] = None
    elapsed_ms: Optional[int] = None
    error: Optional[str] = None


@router.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(req: SynthesizeRequest) -> SynthesizeResponse:
    # Adapter selection is a config-time concern; unrecoverable errors
    # (missing env var, unimplemented provider) surface as 503.
    try:
        adapter = get_llm_adapter()
    except LLMError as e:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "llm_provider_unavailable",
                "message": str(e),
            },
        )

    # synthesize() is sync (adapters use requests/httpx internally); run in
    # a threadpool so we don't block the event loop.
    try:
        result = await asyncio.to_thread(
            adapter.synthesize,
            req.prompt,
            model_hint=req.model_hint,
            max_tokens=req.max_tokens,
        )
    except LLMError as e:
        # Config-level failures that only manifest at call time
        raise HTTPException(
            status_code=503,
            detail={
                "error": "llm_provider_unavailable",
                "message": str(e),
            },
        )

    return SynthesizeResponse(
        text=result.text,
        provider=result.provider,
        model=result.model,
        tokens_used=result.tokens_used,
        neurons_used=result.neurons_used,
        cost_usd=result.cost_usd,
        elapsed_ms=result.elapsed_ms,
        error=result.error,
    )


@router.get("/health")
async def llm_health():
    """Report configured LLM provider status.

    Returns provider slug on success; a hint about the configuration
    problem on unconfigured/unavailable providers. Never raises — this
    endpoint is safe for uptime probes.
    """
    try:
        adapter = get_llm_adapter()
        return {"status": "ok", "provider": adapter.provider_name}
    except LLMError as e:
        return {"status": "unconfigured", "error": str(e)}
    except Exception as e:  # noqa: BLE001 — health probe must not raise
        return {"status": "error", "error": str(e)}
