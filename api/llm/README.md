# `api/llm/` — VPS LLM adapter layer

Pluggable provider architecture so report modules (and future
SQL-interpreter / "Ask this cart" surfaces) can call one method and get
a synthesis result regardless of which LLM backend is configured.

```
llm = get_llm_adapter()
result = llm.synthesize(prompt, model_hint="large", max_tokens=1024)
```

## The three-tier LLM story

Full framing lives in
`docs/vps-internal/Cloudflare Agents Investigation 2026-07-10.md`
(Section 7). Short version:

| Tier             | Provider  | Adapter                     | Cost to VPS                       | Quality              | User setup                       |
|------------------|-----------|-----------------------------|-----------------------------------|----------------------|----------------------------------|
| **Free**         | Track C   | `CloudflareAdapter`         | ~$0 for first ~200 users          | Good for reports     | None (default)                   |
| **Enterprise**   | Track B   | `AnthropicAdapter` *(TBD)*  | Per-token Claude billing          | Best                 | None                             |
| **Power user**   | Track A   | `HeartbeatAdapter` *(TBD)*  | $0                                | Best (their Claude)  | Install Heartbeat + Claude sub   |

Cloudflare Workers AI is the default. Anthropic + Heartbeat are
placeholder implementations that raise `LLMError`; the shape is kept
symmetric so future integration is a file-swap.

## Selecting the provider

`get_llm_adapter()` reads `VECTOR_PLUS_LLM_PROVIDER`:

```bash
export VECTOR_PLUS_LLM_PROVIDER=cloudflare   # default
export VECTOR_PLUS_LLM_PROVIDER=anthropic    # not yet implemented
export VECTOR_PLUS_LLM_PROVIDER=heartbeat    # not yet implemented
```

Unknown values raise `LLMError` at construction time so misconfiguration
surfaces immediately.

## Configuring the Cloudflare adapter (Track C)

Two operating modes controlled by `CF_ENDPOINT_MODE`:

### `direct` (default) — POST to Cloudflare's public REST API

```bash
export CF_ENDPOINT_MODE=direct          # optional, this is the default
export CF_ACCOUNT_ID=<your CF account id>
export CF_API_TOKEN=<your CF Workers AI API token>
```

The adapter calls
`POST https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{model}`
with `Authorization: Bearer {CF_API_TOKEN}`. 30-second timeout.

### `worker` — POST to our CF Worker in front of Workers AI

```bash
export CF_ENDPOINT_MODE=worker
export CF_WORKER_URL=https://<worker-subdomain>.workers.dev
export WORKER_AUTH_TOKEN=<shared secret with the Worker>
```

The adapter calls `POST {CF_WORKER_URL}/synthesize` with
`X-Worker-Auth: {WORKER_AUTH_TOKEN}`. 60-second timeout. The Worker
itself is being built in parallel — until it ships, use `direct` mode.

## Configuring Anthropic (Track B, not yet wired)

When the real implementation lands, the config surface will be:

```bash
export VECTOR_PLUS_LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=<your Anthropic key>
```

Endpoint: `POST https://api.anthropic.com/v1/messages` with
`x-api-key: {ANTHROPIC_API_KEY}`.

## Configuring Heartbeat (Track A, not yet wired)

Requires:

- Heartbeat browser extension installed + running.
- User has a Claude Pro / Team subscription attached to Heartbeat.
- Either a loopback REST endpoint on Heartbeat or an MCP handoff via
  Membot (implementation choice deferred).

## Calling `synthesize` from a report module

```python
from api.llm import get_llm_adapter, LLMError, SynthesisResult


def generate_executive_tldr(cart_path: str, top_passages: list[str]) -> str:
    llm = get_llm_adapter()

    prompt = (
        "You are summarizing this cart for a busy office manager.\n\n"
        f"Passages:\n\n{chr(10).join(top_passages)}\n\n"
        "Produce a 5-bullet synthesis."
    )

    try:
        result: SynthesisResult = llm.synthesize(
            prompt,
            model_hint="large",   # Executive TL;DR wants the best model
            max_tokens=1024,
        )
    except LLMError as exc:
        # Config-level failure (missing token, unknown provider) —
        # surface to the operator via the report metadata.
        return f"[LLM unavailable: {exc}]"

    if result.error:
        # Transient failure (rate limit, timeout, network) — the report
        # module decides whether to retry, fall back to a smaller model,
        # or degrade to a non-LLM Summary Report shape.
        return f"[LLM soft-failed: {result.error}]"

    return result.text
```

## Model-hint reference

Every provider translates the same four hints. If a hint isn't
recognized, providers fall back to `default` silently.

| `model_hint` | Cloudflare (Track C)                       | Anthropic (Track B, planned) |
|--------------|--------------------------------------------|-------------------------------|
| `default`    | `@cf/meta/llama-3.3-70b-instruct-fp8-fast` | `claude-haiku-4-5`            |
| `small`      | `@cf/meta/llama-3.1-8b-instruct`           | `claude-haiku-4-5`            |
| `large`      | `@cf/meta/llama-3.3-70b-instruct-fp8-fast` | `claude-sonnet-5`             |
| `vision`     | `@cf/meta/llama-4-scout-17b-16e-instruct`  | `claude-opus-4-8`             |

Report-type → model-hint recommendation lives in Section 3 of the
investigation doc:

| Report            | Recommended hint |
|-------------------|------------------|
| Summary           | `small`          |
| Timeline          | `small`          |
| Trend             | `small`          |
| Comparison        | `large`          |
| Entity Rollup     | `small`          |
| Financial         | `small`          |
| Change Log        | `large`          |
| Executive TL;DR   | `large`          |

## Adding a new provider

1. Drop `api/llm/providers/newprovider.py` with a class extending
   `LLMAdapter` (see `cloudflare.py` for the canonical shape).
2. Re-export in `api/llm/providers/__init__.py`.
3. Add the string key + constructor call in
   `api/llm/registry.py::get_llm_adapter`.
4. Update this README's provider table + config section.

Provider `__init__` should validate env vars and raise `LLMError` on
missing config — that gets the misconfiguration surfaced at startup
rather than at first synthesis call.

## Error handling contract

- `LLMError` is raised for config-level problems the caller can't fix
  at runtime (missing API key, unknown provider, adapter not
  implemented). Callers should treat this as fatal for the request.
- Transient failures (rate limit, timeout, network blip, non-2xx
  responses other than 401/429) surface via
  `SynthesisResult.error`. `text` is empty in that case; the caller
  decides whether to retry, fall back to a different model_hint, fall
  back to a different provider, or degrade the feature.
- 401 always raises `LLMError` (auth is config, not transient).
- 429 always populates `SynthesisResult.error="rate limited"` — same
  shape as other transient failures so the fallback code path stays
  uniform.
