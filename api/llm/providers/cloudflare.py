"""Cloudflare Workers AI adapter — Track C.

Implements :class:`~api.llm.adapter.LLMAdapter` against Cloudflare's
hosted open-source LLMs (Llama family). This is the default provider for
VPS free-tier customers: ~$0 for the first ~200 users, ~7-10x cheaper
than Anthropic direct at scale.

Two operating modes, selected by the ``CF_ENDPOINT_MODE`` env var:

- ``direct`` (default) — the adapter POSTs directly to Cloudflare's
  public REST API. Requires ``CF_ACCOUNT_ID`` + ``CF_API_TOKEN``.
- ``worker`` — the adapter POSTs to a Cloudflare Worker we run in front
  of Workers AI (adds auth, rate-limiting, request tagging). Requires
  ``CF_WORKER_URL`` + ``WORKER_AUTH_TOKEN``.

The Worker itself is being built in parallel by another agent; the code
path here is just the client side.

See ``docs/vps-internal/Cloudflare Agents Investigation 2026-07-10.md``
Section 3 for model-pick rationale and Section 5 for the two integration
patterns.
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

import httpx

from ..adapter import LLMAdapter, LLMError, SynthesisResult


# ---------------------------------------------------------------------------
# Model translation table
# ---------------------------------------------------------------------------
# model_hint → actual Cloudflare Workers AI model id. Kept in one place so
# a CF model deprecation is a single-line edit (Section 6, Risk #3 in the
# investigation doc).
#
# Rationale for each pick, from Section 3 of the investigation:
#   default / large : Llama 3.3 70B fp8-fast — the workhorse for report
#                     synthesis + function calling. Best quality-per-neuron
#                     in CF's catalog as of 2026-07.
#   small           : Llama 3.1 8B — cheap template-filling for Summary /
#                     Timeline / Trend reports. Marked deprecated by CF but
#                     still the smallest good instruct model available; if
#                     it disappears, bump to Gemma 3 or Llama 3.2 3B.
#   vision          : Llama 4 Scout 17B — has vision + long context; use
#                     for graphic-pattern reports once we wire images
#                     through. If unavailable in the account, the CF API
#                     will 4xx and the adapter will bubble the error up in
#                     SynthesisResult.error. Explicit fallback logic can
#                     land after we see real availability behavior.
# ---------------------------------------------------------------------------
_MODEL_HINT_MAP: dict[str, str] = {
    "default": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "large": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "small": "@cf/meta/llama-3.1-8b-instruct",
    "vision": "@cf/meta/llama-4-scout-17b-16e-instruct",
}


# Cloudflare AI REST base — account-scoped model runner.
_CF_API_BASE = "https://api.cloudflare.com/client/v4/accounts"

# Timeouts differ by mode: direct calls hit CF's edge from our droplet
# (~200-500ms typical per Section 6 of the investigation doc), so 30s is
# generous. Worker calls go through our own edge Worker, which may do
# extra work (caching, tool calls) — 60s buys headroom without being
# forgiving to genuine hangs.
_DIRECT_TIMEOUT_SEC = 30.0
_WORKER_TIMEOUT_SEC = 60.0

# Rough neuron ↔ token conversion. CF doesn't return exact neuron cost
# in the API response, and the number varies by model (a 70B pass costs
# more neurons per token than an 8B pass). This is a rule-of-thumb for
# per-request accounting so operators can budget-track without a paid
# CF plan. Real per-model calibration is post-MVP work; log the raw
# tokens_used alongside so we can back-fit later.
_NEURONS_PER_TOKEN_ESTIMATE = 2


class CloudflareAdapter(LLMAdapter):
    """Cloudflare Workers AI client. Reads env vars at __init__ time so
    a misconfigured environment fails loudly at startup rather than at
    first synthesis call."""

    def __init__(self) -> None:
        self._mode = os.environ.get("CF_ENDPOINT_MODE", "direct").strip().lower()
        if self._mode not in ("direct", "worker"):
            raise LLMError(
                f"CF_ENDPOINT_MODE={self._mode!r} not recognized. "
                f"Expected 'direct' or 'worker'."
            )

        if self._mode == "direct":
            self._account_id = os.environ.get("CF_ACCOUNT_ID", "").strip()
            self._api_token = os.environ.get("CF_API_TOKEN", "").strip()
            if not self._account_id or not self._api_token:
                raise LLMError(
                    "CF_ACCOUNT_ID and CF_API_TOKEN must both be set when "
                    "CF_ENDPOINT_MODE=direct."
                )
            self._worker_url: Optional[str] = None
            self._worker_auth: Optional[str] = None
        else:
            self._worker_url = os.environ.get("CF_WORKER_URL", "").strip()
            self._worker_auth = os.environ.get("WORKER_AUTH_TOKEN", "").strip()
            if not self._worker_url or not self._worker_auth:
                raise LLMError(
                    "CF_WORKER_URL and WORKER_AUTH_TOKEN must both be set when "
                    "CF_ENDPOINT_MODE=worker."
                )
            self._account_id = None  # type: ignore[assignment]
            self._api_token = None   # type: ignore[assignment]

    # ------------------------------------------------------------------
    # LLMAdapter interface
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "cloudflare"

    def synthesize(
        self,
        prompt: str,
        *,
        model_hint: str = "default",
        max_tokens: int = 2048,
    ) -> SynthesisResult:
        # Unknown hints fall back to 'default' silently — the contract in
        # LLMAdapter.synthesize says "unknown hints fall back to 'default'",
        # and callers pass user-configured strings that shouldn't crash
        # the pipeline.
        model = _MODEL_HINT_MAP.get(model_hint, _MODEL_HINT_MAP["default"])

        start = time.monotonic()
        try:
            if self._mode == "direct":
                response_json = self._call_direct(model, prompt, max_tokens)
            else:
                response_json = self._call_worker(model, prompt, max_tokens)
        except LLMError:
            # Config-level failures propagate; caller can't recover.
            raise
        except _RateLimited:
            return SynthesisResult(
                text="",
                provider=self.provider_name,
                model=model,
                elapsed_ms=int((time.monotonic() - start) * 1000),
                error="rate limited",
            )
        except (httpx.HTTPError, TimeoutError, OSError) as exc:
            # Transient network / timeout failures — soft-fail with the
            # exception string in SynthesisResult.error so the caller can
            # retry, fall back to a different provider, or degrade the
            # report gracefully.
            return SynthesisResult(
                text="",
                provider=self.provider_name,
                model=model,
                elapsed_ms=int((time.monotonic() - start) * 1000),
                error=str(exc),
            )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        text, tokens_used = _extract_text_and_tokens(response_json)
        neurons_used = (
            tokens_used * _NEURONS_PER_TOKEN_ESTIMATE
            if tokens_used is not None
            else None
        )

        return SynthesisResult(
            text=text,
            provider=self.provider_name,
            model=model,
            tokens_used=tokens_used,
            neurons_used=neurons_used,
            elapsed_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # HTTP calls — split by mode so the retry / auth boundaries stay clear
    # ------------------------------------------------------------------

    def _call_direct(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        """POST to CF's public REST endpoint.

        Endpoint shape:
            POST https://api.cloudflare.com/client/v4/accounts/{account_id}
                /ai/run/{model}
            Headers: Authorization: Bearer {api_token}
            Body: {"prompt": "...", "max_tokens": N}
        """
        url = f"{_CF_API_BASE}/{self._account_id}/ai/run/{model}"
        headers = {
            "Authorization": f"Bearer {self._api_token}",
            "Content-Type": "application/json",
        }
        payload = {"prompt": prompt, "max_tokens": max_tokens}

        with httpx.Client(timeout=_DIRECT_TIMEOUT_SEC) as client:
            resp = client.post(url, headers=headers, json=payload)

        return self._handle_response(resp)

    def _call_worker(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        """POST to our own CF Worker in front of Workers AI.

        The Worker (built in parallel by another agent) accepts a
        ``model`` + ``prompt`` + ``max_tokens`` JSON body on
        ``/synthesize`` and returns the same shape CF returns from
        ``/ai/run/{model}``. Auth is a shared header, not a Bearer
        token, so the Worker can front-check without hitting CF for
        obviously-bad requests.
        """
        url = f"{self._worker_url.rstrip('/')}/synthesize"
        headers = {
            "X-Worker-Auth": self._worker_auth or "",
            "Content-Type": "application/json",
        }
        payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens}

        with httpx.Client(timeout=_WORKER_TIMEOUT_SEC) as client:
            resp = client.post(url, headers=headers, json=payload)

        return self._handle_response(resp)

    @staticmethod
    def _handle_response(resp: httpx.Response) -> dict[str, Any]:
        """Map an httpx response to either parsed JSON, an LLMError, or
        a transient signal that gets turned into SynthesisResult.error
        upstream."""
        if resp.status_code == 401:
            raise LLMError("Cloudflare auth failed")
        if resp.status_code == 429:
            raise _RateLimited()
        if not resp.is_success:
            # Any other non-2xx: bubble as an httpx.HTTPStatusError so it
            # lands in the transient path (populates SynthesisResult.error
            # rather than raising LLMError). If we ever want distinct
            # handling for 5xx vs 4xx, split here.
            raise httpx.HTTPStatusError(
                f"Cloudflare returned {resp.status_code}: {resp.text[:200]}",
                request=resp.request,
                response=resp,
            )
        return resp.json()


# ---------------------------------------------------------------------------
# Internal signal used to hop from _handle_response back to synthesize()
# without leaking a CF-specific exception type into the public interface.
# ---------------------------------------------------------------------------
class _RateLimited(Exception):
    pass


def _extract_text_and_tokens(
    response_json: dict[str, Any],
) -> tuple[str, Optional[int]]:
    """Pull the completion text and token count out of a CF Workers AI
    response envelope.

    CF's ``/ai/run/{model}`` response shape is:

        {
          "result": {
            "response": "<the text>",
            "usage": {
              "prompt_tokens": N,
              "completion_tokens": M,
              "total_tokens": N + M
            }
          },
          "success": true,
          "errors": [],
          "messages": []
        }

    Some models omit ``usage`` entirely. Missing keys degrade to
    ``text=""`` / ``tokens_used=None`` rather than raising, so callers
    always get a well-formed :class:`SynthesisResult`.
    """
    result = response_json.get("result") or {}
    if not isinstance(result, dict):
        return "", None

    text = result.get("response") or ""
    if not isinstance(text, str):
        text = str(text)

    usage = result.get("usage") or {}
    if isinstance(usage, dict):
        tokens = usage.get("total_tokens")
        if isinstance(tokens, (int, float)):
            return text, int(tokens)

    return text, None
