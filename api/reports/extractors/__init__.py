"""Shared extraction primitives for the VPS Reports engine.

Foundation module. Timeline, Trend, Entity Rollup, and Financial
Rollup — plus any future reports that layer LLM extraction on top —
all consume the three functions and three dataclasses re-exported here.

Pure regex today. An optional LLM-fallback callable per extractor for
ambiguous inputs is a planned extension; see the ``TODO`` markers in
each submodule for where it plugs in.

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
from .entities import MentionSpan, discover_entities, extract_entity_mentions

__all__ = [
    # Functions
    "extract_dates",
    "extract_currency",
    "extract_entity_mentions",
    "discover_entities",
    # Dataclasses
    "DateExtraction",
    "MoneyExtraction",
    "MentionSpan",
]
