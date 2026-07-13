# `api/reports/extractors/` — shared extraction primitives

Wave-1a foundation module for the VPS Reports engine. Provides pure
regex extraction of **dates**, **currency amounts**, and **entity
mentions** from raw passage text. Timeline (§2), Trend (§3), Entity
Rollup (§5), and Financial Rollup (§6) all consume this.

Design source-of-truth: `docs/vps-internal/Report Types Design
2026-07-10.md` §0.3.

## What ships in Wave 1

Pure regex. No LLM calls, no network I/O, no cart-file dependency —
these functions operate on raw strings only. That's deliberate: report
modules construct a `CartHandle`, iterate passages, then call
extractors on each passage's text.

Confidence per extractor:

| Extractor | Wave-1 confidence range | Notes |
|---|---|---|
| `extract_dates` | 0.9-1.0 | ISO = 1.0; slash = 0.9; long/compact = 0.95 |
| `extract_currency` | 0.9-1.0 | `$` prefix = 1.0; `EUR/GBP` = 0.9-0.95 |
| `extract_entity_mentions` | n/a | Boolean match, no confidence field |

## Wave-2 hooks (deferred)

Each extractor has a `TODO(wave-2)` marker documenting where the LLM
fallback plugs in without changing the return shape:

- **Dates**: relative-date resolution ("yesterday", "last Tuesday",
  "next Q3"). Consumes a `(text, anchor_date)` callable, emits
  additional `DateExtraction` records with `confidence < 1.0`.
- **Currency**: ambiguous shapes ("twelve ninety-five", "USD twelve
  fifty", bare-number-in-money-context). Consumes a
  `(text, prior_hits)` callable.
- **Entities**: NER for un-hinted entities. Not called out here — an
  external NER pass upstream of `extract_entity_mentions` is the
  cleaner boundary.

The LLM fallback dispatches through `api/llm/adapter.py` — see
`api/llm/README.md` for the three-tier story (Cloudflare / BYO Claude
/ Heartbeat).

## Public API

```python
from api.reports.extractors import (
    extract_dates,
    extract_currency,
    extract_entity_mentions,
    DateExtraction,
    MoneyExtraction,
    MentionSpan,
)
```

### `extract_dates(text: str) -> list[DateExtraction]`

Wave-1 formats:

- **ISO**: `YYYY-MM-DD` — unambiguous, highest confidence
- **US slash**: `MM/DD/YYYY`, `M/D/YY`, `M/D/YYYY` (2-digit year
  pivots on 70: `26` → 2026, `85` → 1985)
- **Long**: `January 5, 2026`, `Jan 5 2026`, `5 January 2026`,
  ordinal suffixes tolerated (`March 3rd, 2026`)
- **Compact**: `YYYYMMDD` — Sysco-style filename embedding, e.g.
  `20260517`. Year gated to 1900-2099 to prevent junk-digit hits.

Overlapping matches resolved by confidence (higher wins), tie-broken
by longer span.

```python
>>> extract_dates("Delivered 2026-05-17 per invoice 752657234_20260517_034.pdf")
[
    DateExtraction(date=date(2026,5,17), source_format="iso", ...),
    DateExtraction(date=date(2026,5,17), source_format="compact", ...),
]
```

### `extract_currency(text, patterns=None) -> list[MoneyExtraction]`

Wave-1 shapes:

- `$` prefix: `$12.34`, `$1,234.56`, `$12`
- Parenthesized refunds: `($12.34)` — `is_negative=True`
- Trailing currency codes: `12.34 USD` (accepts USD/EUR/GBP/CAD/AUD/
  JPY/CHF)
- Symbol prefix: `€12.34`, `£12.34`

All amounts are `Decimal` — never floats — so upstream aggregators
don't accumulate FP error over thousands of invoice lines.

**`context_hint`**: nearest alphabetic word within 20 chars of the
match (left-preferred). Populated on every hit; downstream reports
use it to route amounts to the right bucket without a second pass.

**Preset filtering**: if `patterns=` is provided, hits are filtered
against a preset library. Wave-1 presets (defined in
`currency.py::_PRESET_TOKENS`):

| Preset | Matches context_hint containing |
|---|---|
| `fuel_surcharge` | `fuel`, `surcharge` |
| `invoice_total` | `total`, `invoice` |
| `state_fee` | `state`, `fee` |
| `tax` | `tax`, `sales` |
| `delivery` | `delivery` |
| `tip` | `tip`, `gratuity` |

Unknown preset names are treated as literal token filters. Wave-2 will
move this to `presets/trend_presets.yaml` per §3 of the design doc.

```python
>>> extract_currency("Fuel surcharge $8.95 and total $234.56", patterns=["fuel_surcharge"])
[MoneyExtraction(value=Decimal("8.95"), currency="USD", context_hint="surcharge", ...)]
```

### `extract_entity_mentions(text, entity_name, aliases=None, context_chars=100) -> list[MentionSpan]`

Case-insensitive substring match. Aliases are additional strings that
count as mentions of the same canonical `entity_name` (returned in
`MentionSpan.entity_name` for grouping; `matched_text` preserves the
surface form).

**Overlap resolution**: when the canonical name and an alias both fire
on the same span (e.g. `"Sysco Portland"` + alias `"Sysco"` firing on
`"Sysco Portland Inc"`), the **longer** surface form wins so the
caller sees the most-specific mention.

**Context extraction**: expands out from the match toward sentence
terminators (`. ! ?` or newline) up to `context_chars` on each side.
Falls back to a raw char slice if no boundary is found in range.

```python
>>> extract_entity_mentions(
...     "Sysco Portland delivered. Sysco truck at 6am. sysco-portland invoiced.",
...     "Sysco Portland",
...     aliases=["Sysco", "sysco-portland"],
... )
[MentionSpan(matched_text="Sysco Portland", ...),
 MentionSpan(matched_text="Sysco", ...),
 MentionSpan(matched_text="sysco-portland", ...)]
```

## Usage in a report module

```python
from api.reports import Report, ReportOutput, register_report
from api.reports.cart_reader import CartHandle
from api.reports.extractors import (
    extract_dates, extract_currency, extract_entity_mentions,
)


@register_report
class FuelSurchargeTrendReport(Report):
    name = "trend"
    llm_dependency = False

    def generate(self, cart_path, inputs, options):
        cart = CartHandle(cart_path)
        series = []
        for idx, text, source in cart.iter_passages():
            money = extract_currency(text, patterns=["fuel_surcharge"])
            if not money:
                continue
            dates = extract_dates(text)
            when = dates[0].date if dates else None
            series.extend((when, m.value, source) for m in money)
        # ...aggregate by month, emit markdown, return ReportOutput
```

## Adding a new extraction pattern

When adding a Wave-2 report or new preset:

1. Prefer extending the existing `_PRESET_TOKENS` map over adding a
   new extractor function. New tokens should describe the invoice /
   passage context (label word), not the amount shape itself.
2. If a genuinely new shape is needed (e.g. cryptocurrency
   `0.00012345 BTC`), add a new regex in `currency.py` beside
   `_TRAILING_CODE_RE`, add a fixture in `test_extractors.py`, and
   surface the new currency code via the returned `currency` field.
3. For a new **date format** (e.g. European `DD.MM.YYYY`), add the
   regex in `dates.py`, wire it into `extract_dates`, and ensure the
   overlap-resolution step still sorts your new format's confidence
   correctly relative to the existing four.
4. Add fixture rows to `test_extractors.py`. Match the "verify by
   running the test file directly" pattern already in place — the
   `_print_sample_outputs` block at the bottom is where new sample
   outputs go for the completion report.

## Running the tests

```bash
# With pytest (preferred):
python -m pytest api/reports/extractors/test_extractors.py -v

# Standalone (also prints sample outputs):
python api/reports/extractors/test_extractors.py
```

The standalone runner prints sample outputs on the Sysco fixtures
before running the unittest suite — handy when triaging a regression
against the demo cart.

## Points at design doc

- §0.3 — the shared-extractor contract (this module)
- §2 (Timeline) — primary consumer of `extract_dates`
- §3 (Trend) — primary consumer of `extract_currency` + preset library
- §5 (Entity Rollup) — primary consumer of `extract_entity_mentions`
- §6 (Financial Rollup) — cross-consumer of currency + entity + dates
- §C.1 — wave-1/wave-2 extraction-quality strategy
