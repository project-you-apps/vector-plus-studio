"""Public surface for the VPS Reports engine.

Report modules and API routes should reach for the symbols re-exported
here rather than importing from the internal submodules. This package
is the base every report subclasses.

Usage — building a new report::

    from api.reports import Report, ReportInput, ReportOptions, \
        ReportOutput, register_report
    from api.reports.cart_reader import CartHandle

    @register_report
    class SummaryReport(Report):
        name = "summary"
        display_name = "Summary"
        description = "Cart orientation."
        input_schema = [...]           # mirror report-definitions.ts
        llm_dependency = False
        supports_scheduling = True

        def generate(self, cart_path, inputs, options):
            cart = CartHandle(cart_path)
            ...
            return ReportOutput(markdown="...", metadata={...})

Usage — dispatching a report from a FastAPI route::

    from api.reports import run_report, ReportOptions

    output = run_report(
        report_name="summary",
        cart_path="/opt/membot/cartridges/foo.cart.npz",
        raw_inputs={"top_themes": 5},
        options=ReportOptions(max_llm_calls=0),
    )
    return output.to_json_response()
"""
from .base import Report, ReportInput, ReportOptions, ReportOutput
from .cart_reader import CartHandle
from .executor import run_report
from .registry import (
    REGISTRY,
    get_report_by_name,
    list_reports,
    register_report,
)

__all__ = [
    # Core types
    "Report",
    "ReportInput",
    "ReportOptions",
    "ReportOutput",
    # Cart access
    "CartHandle",
    # Registry
    "REGISTRY",
    "register_report",
    "get_report_by_name",
    "list_reports",
    # Runner
    "run_report",
]
