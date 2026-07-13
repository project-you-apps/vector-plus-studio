"""Shared extraction primitives for the VPS Reports engine.

Wave-1a foundation module (Andy 2026-07-11). Timeline (§2), Trend (§3),
Entity Rollup (§5), and Financial Rollup (§6) — plus any wave-2 reports
that layer LLM extraction on top — all consume the three functions and
three dataclasses re-exported here.

Design: ``docs/vps-internal/Report Types Design 2026-07-10.md`` §0.3.
This wave: pure regex. Wave-2 hook: an optional LLM-fallback callable
per extractor for ambiguous inputs; see the ``TODO(wave-2)`` markers
in each submodule.

Usage::

    from api.reports.extractors import (
        extract_dates, extract_currency, extract_entity_mentions,
        DateExtraction, MoneyExtraction, MentionSpan,
    )

    for _, text, _ in cart.iter_passages():
        dates = extract_dates(text)
        money = extract_currency(text, patterns=["fuel_surcharge"])
        mentions = extract_entity_mentions(
            text, "Sysco Portland",
            aliases=["Sysco", "SYSCO PORTLAND"],
        )
"""
from .currency import MoneyExtraction, extract_currency
from .dates import DateExtraction, extract_dates
from .entities import MentionSpan, extract_entity_mentions

__all__ = [
    # Functions
    "extract_dates",
    "extract_currency",
    "extract_entity_mentions",
    # Dataclasses
    "DateExtraction",
    "MoneyExtraction",
    "MentionSpan",
]
