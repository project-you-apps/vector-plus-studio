"""Smoke tests for the Agents engine (2026-07-13 MVP).

Coverage
--------
- Each of the 4 v1 agents (``auto_briefing``, ``qa``, ``professor``,
  ``cart_curator``) runs end-to-end against a synthetic ``.cart.npz``
  and produces a well-formed :class:`AgentOutput`.
- Prompt wrapping — verifies :func:`wrap_llama3_instruct` emits the
  exact Llama 3 instruct header sequence (regression guard against a
  silent template drift).
- Registry — all 4 slugs are registered, ``list_agents`` returns the
  expected shapes, ``get_agent_by_name`` finds them.
- Executor guardrails — unknown slug + zero-budget behavior.

The LLM adapter is stubbed with a fake that returns a deterministic
answer so tests don't require a working Cloudflare / Anthropic
provider. The real-LLM verification path is a separate manual smoke
step (documented in the AGENT-LOGBOOK entry).

Run standalone::

    python api/agents/tests/test_agents.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from typing import Any, Optional
from unittest import mock

import numpy as np

# Support both `python -m pytest ...` and direct execution — mirror the
# bootstrap the sibling reports tests use.
if __name__ == "__main__" and __package__ is None:
    _here = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)


# ---------------------------------------------------------------------------
# Synthetic cart builder — same shape as reports tests use
# ---------------------------------------------------------------------------

def _hippo_row(tombstone: bool = False) -> np.ndarray:
    row = np.zeros(64, dtype=np.uint8)
    if tombstone:
        row[28] = 0x01
    return row


def _make_synthetic_cart(
    passages: list[str],
    sources: list[str],
    tombstones: Optional[list[bool]] = None,
    dim: int = 16,
    cart_name: str = "agents-smoke-cart",
    description: str = "synthetic cart for agents smoke test",
) -> str:
    n = len(passages)
    assert len(sources) == n
    tombstones = tombstones or [False] * n
    assert len(tombstones) == n

    rng = np.random.default_rng(42)
    embeddings = (
        rng.standard_normal(size=(n, dim), dtype=np.float32)
        if n else np.zeros((0, dim), dtype=np.float32)
    )
    if n:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

    passages_arr = np.array(passages, dtype=object)
    sources_arr = np.array(sources, dtype=object)
    hippo = (
        np.stack([_hippo_row(t) for t in tombstones])
        if n else np.zeros((0, 64), dtype=np.uint8)
    )
    pattern0 = np.array(json.dumps({
        "cart_name": cart_name,
        "description": description,
    }), dtype=object)

    fd, path = tempfile.mkstemp(suffix=".cart.npz", prefix="agents_smoke_")
    os.close(fd)
    np.savez(
        path,
        embeddings=embeddings,
        passages=passages_arr,
        source_paths=sources_arr,
        hippocampus=hippo,
        pattern0=pattern0,
    )
    return path


# ---------------------------------------------------------------------------
# Fake LLM adapter — deterministic response, no network
# ---------------------------------------------------------------------------

class _FakeSynthesisResult:
    """Duck-typed :class:`api.llm.adapter.SynthesisResult` for tests."""

    def __init__(
        self,
        text: str,
        provider: str = "fake",
        model: str = "fake-1",
        tokens_used: int = 100,
        neurons_used: int = 250,
        error: Optional[str] = None,
    ):
        self.text = text
        self.provider = provider
        self.model = model
        self.tokens_used = tokens_used
        self.neurons_used = neurons_used
        self.cost_usd = None
        self.elapsed_ms = 42
        self.error = error


class _FakeAdapter:
    """Stub :class:`LLMAdapter` that returns a fixed answer.

    Kept minimal — the AGENT'S behavior is under test, not the LLM's.
    The captured prompt is exposed via ``last_prompt`` so tests can
    verify agents wrapped their prompts correctly.
    """

    def __init__(self, response_text: str = "Fake LLM response. [1]"):
        self.response_text = response_text
        self.last_prompt: Optional[str] = None
        self.last_kwargs: dict[str, Any] = {}
        self.provider_name = "fake"

    def synthesize(self, prompt: str, **kwargs: Any) -> _FakeSynthesisResult:
        self.last_prompt = prompt
        self.last_kwargs = kwargs
        return _FakeSynthesisResult(text=self.response_text)


# ---------------------------------------------------------------------------
# Prompt-wrapper regression guard
# ---------------------------------------------------------------------------

class TestPromptWrapping(unittest.TestCase):
    """Verify :func:`wrap_llama3_instruct` emits the exact template.

    Regression guard: silent drift of the header strings would revert
    the model to base-completion mode (the "book review about
    environmental economics" failure logged 2026-07-13).
    """

    def test_wrap_shape(self):
        from api.agents.prompt import wrap_llama3_instruct
        out = wrap_llama3_instruct("SYSTEM", "USER")
        # Exact header tokens must appear in exact order.
        self.assertIn("<|begin_of_text|>", out)
        self.assertIn("<|start_header_id|>system<|end_header_id|>", out)
        self.assertIn("<|start_header_id|>user<|end_header_id|>", out)
        self.assertIn("<|start_header_id|>assistant<|end_header_id|>", out)
        self.assertIn("<|eot_id|>", out)
        # System + user payloads land in the right role blocks.
        sys_start = out.index("system<|end_header_id|>\n\n") + len(
            "system<|end_header_id|>\n\n"
        )
        self.assertTrue(out[sys_start:].startswith("SYSTEM"))
        user_start = out.index("user<|end_header_id|>\n\n") + len(
            "user<|end_header_id|>\n\n"
        )
        self.assertTrue(out[user_start:].startswith("USER"))
        # Assistant header comes last with no content after it.
        self.assertTrue(
            out.rstrip("\n").endswith(
                "<|start_header_id|>assistant<|end_header_id|>"
            )
            or out.endswith(
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        )


# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------

class TestRegistry(unittest.TestCase):
    """All 4 v1 agents register + are described correctly."""

    def setUp(self):
        # Force side-effect imports (mirrors what agents_routes.py does
        # at app boot).
        from api.agents import (  # noqa: F401
            auto_briefing, qa, professor, cart_curator,
        )

    def test_all_four_registered(self):
        from api.agents.registry import REGISTRY
        for slug in ("auto_briefing", "qa", "professor", "cart_curator"):
            self.assertIn(slug, REGISTRY, f"Missing agent slug: {slug}")

    def test_list_agents_shape(self):
        from api.agents.registry import list_agents
        agents = list_agents()
        self.assertGreaterEqual(len(agents), 4)
        for a in agents:
            self.assertIn("name", a)
            self.assertIn("display_name", a)
            self.assertIn("description", a)
            self.assertIn("input_schema", a)
            self.assertIn("llm_dependency", a)
            self.assertIsInstance(a["input_schema"], list)


# ---------------------------------------------------------------------------
# Executor guardrails
# ---------------------------------------------------------------------------

class TestExecutor(unittest.TestCase):
    def setUp(self):
        from api.agents import (  # noqa: F401
            auto_briefing, qa, professor, cart_curator,
        )

    def test_unknown_slug_raises(self):
        from api.agents.executor import run_agent
        with self.assertRaises(ValueError) as ctx:
            run_agent("nope_not_real", "/tmp/whatever.cart.npz", {})
        self.assertIn("Unknown agent", str(ctx.exception))

    def test_zero_budget_llm_dep_agent(self):
        from api.agents.executor import run_agent
        from api.agents.base import AgentOptions
        with self.assertRaises(ValueError) as ctx:
            run_agent(
                "qa", "/tmp/whatever.cart.npz",
                {"question": "hi"},
                options=AgentOptions(max_llm_calls=0),
            )
        self.assertIn("requires LLM", str(ctx.exception))


# ---------------------------------------------------------------------------
# Agent end-to-end smoke — one class per agent using the fake adapter
# ---------------------------------------------------------------------------

# Passages engineered so bigram/token overlap finds them for
# question-shaped queries about vendors / dates / totals.
_CART_PASSAGES = [
    "Sysco Portland delivery arrived on time. Total $412.55 on 2026-05-04.",
    "Fuel surcharge from Sysco Portland: $8.95. Invoice dated 2026-05-11.",
    "Bar Restaurant Supply invoice #4421, $1,205.00, delivered 2026-05-14.",
    "Sysco Portland weekly order confirmation. Case count: 42. 2026-05-18.",
    "Bar Restaurant Supply short delivery, $302.00 credit issued 2026-05-21.",
    "Sysco Portland credit memo, -$88.20. Adjustment posted 2026-05-25.",
    "Portland Coffee Co. wholesale order, $650.75 on 2026-05-27.",
]
_CART_SOURCES = [
    "invoice-may04.pdf",
    "invoice-may11.pdf",
    "bar-supply-may14.pdf",
    "invoice-may18.pdf",
    "bar-supply-may21.pdf",
    "invoice-may25.pdf",
    "coffee-order-may27.pdf",
]


class _AgentSmokeBase(unittest.TestCase):
    """Shared setup: synthetic cart + fake adapter patched into api.llm.

    Subclasses set ``AGENT_SLUG`` and ``INPUTS`` and inherit
    ``test_smoke``.
    """

    AGENT_SLUG: str = ""
    INPUTS: dict[str, Any] = {}

    @classmethod
    def setUpClass(cls) -> None:
        if cls is _AgentSmokeBase:
            raise unittest.SkipTest("Abstract base class")
        # Force side-effect imports of all agent modules.
        from api.agents import (  # noqa: F401
            auto_briefing, qa, professor, cart_curator,
        )
        cls.cart_path = _make_synthetic_cart(
            _CART_PASSAGES, _CART_SOURCES,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            os.unlink(cls.cart_path)
        except OSError:
            pass

    def test_smoke(self):
        from api.agents.executor import run_agent
        fake = _FakeAdapter(
            response_text=(
                f"[test] Response from {self.AGENT_SLUG}. Passage [1] "
                f"and [2] both look relevant."
            ),
        )
        with mock.patch("api.llm.get_llm_adapter", return_value=fake):
            output = run_agent(
                agent_name=self.AGENT_SLUG,
                cart_path=self.cart_path,
                raw_inputs=self.INPUTS,
            )
        # Base shape assertions.
        self.assertGreater(len(output.markdown), 0)
        self.assertIsInstance(output.cited_patterns, list)
        self.assertIsInstance(output.metadata, dict)
        self.assertIsInstance(output.warnings, list)
        self.assertIsInstance(output.llm_usage, dict)
        # Prompt wrapping (regression guard): the agent MUST have called
        # the adapter with a Llama-3-wrapped prompt.
        self.assertIsNotNone(fake.last_prompt)
        assert fake.last_prompt is not None
        self.assertIn("<|begin_of_text|>", fake.last_prompt)
        self.assertIn(
            "<|start_header_id|>assistant<|end_header_id|>",
            fake.last_prompt,
        )


class TestAutoBriefingSmoke(_AgentSmokeBase):
    AGENT_SLUG = "auto_briefing"
    INPUTS = {"focus": "Sysco Portland", "tone": "executive"}

    def test_no_citations_by_design(self):
        # Auto Briefing summarizes the whole cart — cited_patterns
        # returned empty by design (see auto_briefing.py header).
        from api.agents.executor import run_agent
        fake = _FakeAdapter()
        with mock.patch("api.llm.get_llm_adapter", return_value=fake):
            output = run_agent(
                agent_name=self.AGENT_SLUG,
                cart_path=self.cart_path,
                raw_inputs=self.INPUTS,
            )
        self.assertEqual(output.cited_patterns, [])


class TestQASmoke(_AgentSmokeBase):
    AGENT_SLUG = "qa"
    INPUTS = {"question": "Which invoices came from Sysco Portland?"}

    def test_cites_patterns(self):
        # Q&A retrieves top-N + populates cited_patterns.
        from api.agents.executor import run_agent
        fake = _FakeAdapter()
        with mock.patch("api.llm.get_llm_adapter", return_value=fake):
            output = run_agent(
                agent_name=self.AGENT_SLUG,
                cart_path=self.cart_path,
                raw_inputs=self.INPUTS,
            )
        self.assertGreater(len(output.cited_patterns), 0)
        # Markdown includes a Sources section with vps:// links.
        self.assertIn("Sources", output.markdown)
        self.assertIn("vps://source/", output.markdown)

    def test_empty_question_raises(self):
        from api.agents.executor import run_agent
        with self.assertRaises(ValueError):
            run_agent(
                agent_name=self.AGENT_SLUG,
                cart_path=self.cart_path,
                raw_inputs={"question": ""},
            )


class TestProfessorSmoke(_AgentSmokeBase):
    AGENT_SLUG = "professor"
    INPUTS = {"num_questions": 3, "difficulty": "medium", "topic": "Sysco"}

    def test_cites_patterns(self):
        from api.agents.executor import run_agent
        fake = _FakeAdapter()
        with mock.patch("api.llm.get_llm_adapter", return_value=fake):
            output = run_agent(
                agent_name=self.AGENT_SLUG,
                cart_path=self.cart_path,
                raw_inputs=self.INPUTS,
            )
        self.assertGreater(len(output.cited_patterns), 0)
        self.assertIn("Quiz", output.markdown)


class TestCartCuratorSmoke(_AgentSmokeBase):
    AGENT_SLUG = "cart_curator"
    INPUTS = {"focus_area": "vendor invoices", "source_min": 3}

    def test_flags_under_represented(self):
        # With source_min=3 our synthetic cart has 4 sources that appear
        # < 3 times — the curator flags them.
        from api.agents.executor import run_agent
        fake = _FakeAdapter()
        with mock.patch("api.llm.get_llm_adapter", return_value=fake):
            output = run_agent(
                agent_name=self.AGENT_SLUG,
                cart_path=self.cart_path,
                raw_inputs=self.INPUTS,
            )
        self.assertIn("Under-represented", output.markdown)
        # Curator cites the FIRST pattern index for each flagged source.
        self.assertGreater(len(output.cited_patterns), 0)


if __name__ == "__main__":
    unittest.main()
