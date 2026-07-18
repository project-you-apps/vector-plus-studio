# `api/reports/` — VPS Reports engine

Foundation module for the 8 generic report types. This package is the
base every report subclasses; extractors + Summary / Comparison /
Entity Rollup / Change Log modules dispatch on top of it.

## Wire-up: how a new report gets registered

Report modules subclass `Report`, set the class-level metadata to match
their `report-definitions.ts` entry, and stack `@register_report`:

```python
from api.reports import Report, ReportOutput, register_report
from api.reports.cart_reader import CartHandle


@register_report
class SummaryReport(Report):
    name = "summary"                    # MUST match frontend slug
    display_name = "Summary"
    description = "Cart orientation — what's in this cart, top themes, sources."
    input_schema = [
        {"name": "top_themes", "label": "Top themes", "type": "number",
         "required": False, "default": 5},
    ]
    llm_dependency = False
    supports_scheduling = True

    def generate(self, cart_path, inputs, options):
        cart = CartHandle(cart_path)
        top_themes = inputs.get_int("top_themes", 5)
        # ...produce markdown...
        return ReportOutput(
            markdown=f"# {cart.cart_name}\n\n**{cart.count}** patterns\n...",
            metadata={"unique_sources": len(cart.unique_sources())},
        )
```

Register the module by importing it once in `api/reports/__init__.py`
(add to the `from .modules_summary import *` block that a future version will
introduce). Duplicate `name` values raise at import time.

## Slug naming (frontend ↔ backend contract)

`Report.name` values MUST match the frontend `report-definitions.ts`
entries. As of 2026-07-11 those are:

| Slug | Report |
|---|---|
| `summary` | Summary |
| `timeline` | Timeline |
| `trend` | Trend |
| `comparison` | Comparison |
| `entity_rollup` | Entity Rollup |
| `financial_rollup` | Financial Rollup |
| `change_log` | Change Log |
| `tldr` | Executive TL;DR |

Underscores, not hyphens. the current brief hyphenated some
slugs (e.g. `entity-rollup`); the frontend uses underscores and that's
the source of truth.

## LLM adapter integration

Reports that need synthesis (currently only Executive TL;DR) set
`llm_dependency = True` and call `get_llm_adapter()` inside `generate`:

```python
from api.llm import get_llm_adapter, LLMError, SynthesisResult

class TldrReport(Report):
    name = "tldr"
    llm_dependency = True
    supports_scheduling = True
    # ...

    def generate(self, cart_path, inputs, options):
        llm = get_llm_adapter()
        result = llm.synthesize(
            prompt,
            model_hint=options.llm_model_hint,   # respect caller override
            max_tokens=1024,
        )
        if result.error:
            return ReportOutput(
                markdown="_LLM soft-failed — see warnings._",
                warnings=[f"LLM error: {result.error}"],
            )
        return ReportOutput(markdown=result.text, metadata={
            "llm_calls_made": 1,
            "provider": result.provider,
            "model": result.model,
        })
```

The executor blocks LLM-dependent reports when `options.max_llm_calls`
is zero, so the report code path never runs unless the caller
explicitly opted in.

See `api/llm/README.md` for the three-tier LLM story (Cloudflare / BYO
Claude / Heartbeat) and the model-hint reference table.

## CartHandle usage

`CartHandle` is the shared substrate for reading `.cart.npz` files.
Reports open it once at the top of `generate()`:

```python
cart = CartHandle(cart_path)

# Top-of-cart metadata
cart.cart_name         # filename stem
cart.count             # number of patterns
cart.pattern0          # dict from Pattern-0 JSON (or None)

# Per-pattern access
cart.get_passage(i)    # raw text
cart.get_source(i)     # source filename (falls back to passage line 1)
cart.get_meta(i)       # per_pattern_meta record (image_b64, tags, etc.)

# Bulk enumeration
for idx, text, source in cart.iter_passages():
    ...

cart.unique_sources()  # deduped list of source filenames
cart.embeddings        # raw float32 array for semantic diff (Change Log)
```

Loaded once, cached on the handle, safe for concurrent read-only access
across threads. Don't reach into `_passages` etc. directly — use the
accessors so shape changes stay backwards-compatible.

## Options + input contract

`ReportInput.raw` is the untyped dict from the frontend form. Reports
should read via `get_str` / `get_int` / `get_bool` / `get_list` so
empty-string form submits get treated as "missing":

```python
top_themes = inputs.get_int("top_themes", 5)
aliases = inputs.get_list("aliases")   # comma-split for string forms
```

`ReportOptions` is caller-supplied (NOT user-supplied). Defaults:

- `llm_provider = "cloudflare"` — overrides `VECTOR_PLUS_LLM_PROVIDER`
- `llm_model_hint = "default"` — overrides adapter default
- `max_llm_calls = 0` — hard cap; LLM-dep reports need this ≥1
- `verbose = False` — surfaces options in metadata for audit
- `include_source_refs = True` — pattern-idx refs in metadata

## Testing pattern for report authors

Report modules should ship a smoke test at
`api/reports/tests/test_<report>.py`:

```python
def test_summary_smoke():
    cart_path = "path/to/test_cart.cart.npz"     # real cart on disk
    output = run_report(
        "summary",
        cart_path,
        raw_inputs={"top_themes": 5},
    )
    assert "# " in output.markdown                # has H1
    assert output.metadata["report_name"] == "summary"
    assert output.metadata["elapsed_ms"] < 5000   # <5s
```

`CartHandle` opens files on the local filesystem so tests can be
hermetic against small fixtures in `api/reports/tests/fixtures/`.

## References

- Cart NPZ format: `docs/PATTERN-ANATOMY.md` (H-row), `api/cartbuilder/builder.py` (writer)
