"""Top-level runner for the Reports engine.

This module is the composition point wired to the FastAPI route (Wave-2).
Callers pass a report slug + raw form inputs + optional execution
options and get back a fully-populated :class:`ReportOutput`.

Guardrails enforced here (so no report module has to re-implement them):

- Unknown report slug → :class:`ValueError` with an ``'unknown report'``
  message. The FastAPI route converts this into a 400.
- Report has ``llm_dependency=True`` but ``options.max_llm_calls == 0``
  → :class:`ValueError` pointing at the field. The frontend never sends
  this shape, but the CLI runner + scheduled-brief worker might, and
  the "why did I get a rate-limit error" report is more useful when the
  guardrail message names the exact field to bump.
- Any exception inside ``generate()`` is re-raised with report name +
  cart path context added to the message. The original traceback stays
  attached via ``raise X from exc``.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from .base import ReportInput, ReportOptions, ReportOutput
from .registry import get_report_by_name


def run_report(
    report_name: str,
    cart_path: str,
    raw_inputs: dict[str, Any],
    options: Optional[ReportOptions] = None,
) -> ReportOutput:
    """Look up ``report_name``, instantiate, generate, return output.

    Parameters
    ----------
    report_name
        Slug matching a registered :class:`Report`. Case-sensitive.
    cart_path
        Absolute path to the ``.cart.npz`` file the report reads from.
        The report's :meth:`Report.generate` opens it via
        :class:`~api.reports.cart_reader.CartHandle`.
    raw_inputs
        Untyped dict of user-supplied form values. Wrapped in a
        :class:`ReportInput` for the report to read via the ``get_*``
        accessors.
    options
        Execution options (LLM budget, verbosity, source refs). If
        ``None``, defaults from :class:`ReportOptions` are used.

    Returns
    -------
    :class:`ReportOutput`
        Report's markdown + optional data / warnings / metadata.

    Raises
    ------
    ValueError
        Unknown report slug, or LLM-required report called with a zero
        budget, or the report itself raised anything derivable from
        ValueError.
    Exception
        Whatever the report itself raised — re-raised with report name
        + cart path context added to the message.
    """
    if options is None:
        options = ReportOptions()

    report_cls = get_report_by_name(report_name)
    if report_cls is None:
        # Message shape matches what the frontend/list_reports response
        # displays — helps operators debug the "typo in slug" case fast.
        raise ValueError(
            f"Unknown report: {report_name!r}. Registered reports: "
            f"{sorted(_registered_slugs())}"
        )

    # LLM budget guardrail. Reports that need LLM but got a zero budget
    # would fail deep inside generate() with a less-helpful message;
    # short-circuit here so the error names the actual field to bump.
    if report_cls.llm_dependency and options.max_llm_calls <= 0:
        raise ValueError(
            f"Report {report_name!r} requires LLM synthesis but "
            f"options.max_llm_calls={options.max_llm_calls}. Set "
            f"options.max_llm_calls >= 1 to run this report."
        )

    inputs = ReportInput(raw=dict(raw_inputs or {}))
    report = report_cls()

    t0 = time.time()
    try:
        output = report.generate(cart_path, inputs, options)
    except Exception as exc:
        # Re-raise with context so operator sees which report + cart hit
        # the failure without paging through a bare traceback. Original
        # traceback stays chained via ``from exc``.
        raise type(exc)(
            f"[reports.executor] {report_name!r} against {cart_path!r} "
            f"failed: {exc}"
        ) from exc
    elapsed_ms = int((time.time() - t0) * 1000)

    # Fold execution metadata into the report's own metadata dict. Report
    # modules can pre-populate their own keys; we only augment.
    if not isinstance(output, ReportOutput):
        raise TypeError(
            f"Report {report_name!r} returned {type(output).__name__} "
            f"but Report.generate() must return ReportOutput"
        )

    meta = dict(output.metadata) if output.metadata else {}
    meta.setdefault("report_name", report_name)
    meta.setdefault("cart_path", cart_path)
    meta.setdefault("elapsed_ms", elapsed_ms)
    if options.verbose:
        # Verbose surfaces the options themselves — useful for the UI's
        # "how did this get produced" audit footer.
        meta.setdefault(
            "options",
            {
                "llm_provider": options.llm_provider,
                "llm_model_hint": options.llm_model_hint,
                "max_llm_calls": options.max_llm_calls,
                "include_source_refs": options.include_source_refs,
            },
        )
    output.metadata = meta

    return output


def _registered_slugs() -> list[str]:
    """Return the sorted list of registered report slugs.

    Split out so tests can monkey-patch it if needed. Not part of the
    public surface — use ``list_reports()`` from the registry module.
    """
    # Lazy import to avoid a hard cycle if executor.py is imported before
    # any report module registers.
    from .registry import REGISTRY
    return list(REGISTRY.keys())


__all__ = ["run_report"]
