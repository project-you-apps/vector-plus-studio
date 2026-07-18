"""Smoke tests for POST /api/reports/generate.

Verifies the route contract landed 2026-07-13 (a future release dispatch spine):

- All 5 registered slugs (summary, entity_rollup, change_log, comparison,
  coverage) return HTTP 200 + non-empty markdown.
- Comparison accepts cart_ref_b as advisory metadata (does not error
  either way given current subset-query behavior).
- a future release known slugs (timeline, trend, financial_rollup, tldr) return
  HTTP 501 with the "future release" hint body.
- Unknown slugs return HTTP 404.
- Missing cart_ref returns HTTP 422 (Pydantic).
- Bogus cart_ref returns HTTP 404 with the "cart_not_found" error tag.

Uses fastapi.testclient.TestClient. A synthetic ``.cart.npz`` is
materialized in a tempdir + made discoverable by patching the reports
route's cart resolver to look there — no reliance on the production
``cartridges/`` directory.

Run standalone::

    python api/reports/tests/test_generate_route.py

Or via pytest::

    python -m pytest api/reports/tests/test_generate_route.py -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from typing import Optional

import numpy as np

# Support both `python -m pytest ...` and direct execution — mirror the
# bootstrap the sibling test_coverage.py uses.
if __name__ == "__main__" and __package__ is None:
    _here = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)


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
    cart_name: str = "smoke-test-cart",
) -> str:
    """Materialize a synthetic ``.cart.npz`` and return its abs path.

    Same shape as the sibling coverage-test builder — plain-text
    passages + per-pattern source_paths + a valid hippocampus block.
    Pattern-0 is a JSON string so CartHandle decodes it as a dict.
    """
    n = len(passages)
    assert len(sources) == n
    tombstones = tombstones or [False] * n
    assert len(tombstones) == n

    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal(size=(n, dim), dtype=np.float32) if n else np.zeros((0, dim), dtype=np.float32)
    # Normalize so cosine-based reports have finite similarities.
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
        "description": "synthetic cart for /api/reports/generate smoke test",
    }), dtype=object)

    fd, path = tempfile.mkstemp(suffix=".cart.npz", prefix="reports_route_smoke_")
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


class TestReportsGenerateRoute(unittest.TestCase):
    """End-to-end smoke tests through fastapi.testclient.TestClient."""

    # A cart big enough to exercise every report:
    #  - Sysco Portland mentions for entity_rollup
    #  - Multiple sources for summary
    #  - Currency + date content so extractors have signal for comparison
    CART_PASSAGES = [
        "Sysco Portland delivery arrived on time. Total $412.55 on 2026-05-04.",
        "Fuel surcharge from Sysco Portland: $8.95. Invoice dated 2026-05-11.",
        "Bar Restaurant Supply invoice #4421, $1,205.00, delivered 2026-05-14.",
        "Sysco Portland weekly order confirmation. Case count: 42. 2026-05-18.",
        "Bar Restaurant Supply short delivery, $302.00 credit issued 2026-05-21.",
        "Sysco Portland credit memo, -$88.20. Adjustment posted 2026-05-25.",
        "Portland Coffee Co. wholesale order, $650.75 on 2026-05-27.",
    ]
    CART_SOURCES = [
        "invoice-may04.pdf",
        "invoice-may11.pdf",
        "bar-supply-may14.pdf",
        "invoice-may18.pdf",
        "bar-supply-may21.pdf",
        "invoice-may25.pdf",
        "coffee-order-may27.pdf",
    ]

    @classmethod
    def setUpClass(cls) -> None:
        cls.cart_path = _make_synthetic_cart(
            cls.CART_PASSAGES, cls.CART_SOURCES,
            cart_name="reports-route-smoke",
        )
        # Patch the cart resolver so bare "reports-route-smoke" resolves
        # to our tempfile. Simpler than dropping the file into the real
        # cartridges/ dir and risking cross-test contamination.
        #
        # 2026-07-13: resolver signature changed from ``Optional[str]`` to
        # ``ResolvedCart`` dataclass so the endpoint can distinguish
        # cart_not_found / cart_legacy_format / sandbox_cart_expired.
        # Patch returns the new shape.
        from api import reports_routes as rr
        cls._orig_resolver = rr._resolve_cart_ref

        def _test_resolver(cart_ref: str) -> "rr.ResolvedCart":
            if not cart_ref:
                return rr.ResolvedCart(failure="cart_not_found")
            ref = cart_ref.strip()
            if ref.startswith("local:"):
                return rr.ResolvedCart(failure="local_cart_unsupported")
            if ref.startswith("server:"):
                ref = ref[len("server:"):]
            if ref in ("reports-route-smoke", "reports-route-smoke.cart.npz",
                       os.path.basename(cls.cart_path), cls.cart_path):
                return rr.ResolvedCart(path=cls.cart_path, location="canonical")
            return rr.ResolvedCart(failure="cart_not_found")

        rr._resolve_cart_ref = _test_resolver  # type: ignore[assignment]

        # Build the TestClient AFTER patching so nothing races through
        # the real DATA_DIR-anchored resolver during app startup.
        from fastapi.testclient import TestClient
        from api.main import app
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        from api import reports_routes as rr
        rr._resolve_cart_ref = cls._orig_resolver  # type: ignore[assignment]
        try:
            os.unlink(cls.cart_path)
        except OSError:
            pass

    # -- happy paths ------------------------------------------------------

    def _post_generate(self, payload: dict) -> tuple[int, dict]:
        r = self.client.post("/api/reports/generate", json=payload)
        try:
            body = r.json()
        except Exception:
            body = {"_raw": r.text}
        return r.status_code, body

    def test_summary(self):
        status, body = self._post_generate({
            "report_slug": "summary",
            "cart_ref": "reports-route-smoke",
            "inputs": {"top_themes": 3},
        })
        self.assertEqual(status, 200, body)
        self.assertIn("markdown", body)
        self.assertGreater(len(body["markdown"]), 0)
        self.assertEqual(body["report_slug"], "summary")
        self.assertIn("generated_at", body)

    def test_entity_rollup(self):
        status, body = self._post_generate({
            "report_slug": "entity_rollup",
            "cart_ref": "reports-route-smoke",
            "inputs": {
                "entity_name": "Sysco Portland",
                "aliases": "Sysco PDX",
            },
        })
        self.assertEqual(status, 200, body)
        self.assertGreater(len(body["markdown"]), 0)

    def test_coverage(self):
        status, body = self._post_generate({
            "report_slug": "coverage",
            "cart_ref": "reports-route-smoke",
            "inputs": {},
        })
        self.assertEqual(status, 200, body)
        self.assertIn("Coverage Report", body["markdown"])

    def test_change_log(self):
        # Change Log needs two carts. We hand it the same cart on both
        # sides — the report should render a "no changes" body without
        # crashing. That's enough for the route-level smoke pass.
        status, body = self._post_generate({
            "report_slug": "change_log",
            "cart_ref": "reports-route-smoke",
            "inputs": {
                "cart_id_old": self.cart_path,
                "cart_id_new": self.cart_path,
                "diff_strategy": "exact",
            },
        })
        self.assertEqual(status, 200, body)
        self.assertGreater(len(body["markdown"]), 0)

    def test_comparison(self):
        status, body = self._post_generate({
            "report_slug": "comparison",
            "cart_ref": "reports-route-smoke",
            "inputs": {
                "subset_a_name": "Sysco",
                "subset_a_query": "sysco",
                "subset_b_name": "Bar Supply",
                "subset_b_query": "bar restaurant supply",
            },
        })
        self.assertEqual(status, 200, body)
        self.assertGreater(len(body["markdown"]), 0)

    def test_comparison_with_cart_ref_b_is_advisory(self):
        # cart_ref_b passes through as metadata but does not gate 422 —
        # the actual ComparisonReport uses subset queries only. This
        # documents the intentional deviation from the brief's "required"
        # phrasing (see reports_routes.py module docstring).
        status, body = self._post_generate({
            "report_slug": "comparison",
            "cart_ref": "reports-route-smoke",
            "cart_ref_b": "reports-route-smoke",
            "inputs": {
                "subset_a_name": "Sysco",
                "subset_a_query": "sysco",
                "subset_b_name": "Bar Supply",
                "subset_b_query": "bar restaurant supply",
            },
        })
        self.assertEqual(status, 200, body)
        self.assertEqual(body["metadata"].get("cart_ref_b"), "reports-route-smoke")

    # -- error paths ------------------------------------------------------

    def test_wave2_slug_returns_501(self):
        for slug in ("timeline", "trend", "financial_rollup", "tldr"):
            status, body = self._post_generate({
                "report_slug": slug,
                "cart_ref": "reports-route-smoke",
                "inputs": {},
            })
            self.assertEqual(status, 501, (slug, body))
            self.assertIn("future release", json.dumps(body).lower(), (slug, body))

    def test_unknown_slug_returns_404(self):
        status, body = self._post_generate({
            "report_slug": "not_a_real_report",
            "cart_ref": "reports-route-smoke",
            "inputs": {},
        })
        self.assertEqual(status, 404, body)
        # detail contains the "unknown_report" tag.
        self.assertIn("unknown_report", json.dumps(body))

    def test_bogus_cart_ref_returns_404(self):
        status, body = self._post_generate({
            "report_slug": "summary",
            "cart_ref": "definitely-not-a-real-cart",
            "inputs": {},
        })
        self.assertEqual(status, 404, body)
        self.assertIn("cart_not_found", json.dumps(body))

    def test_local_cart_returns_404_with_local_tag(self):
        status, body = self._post_generate({
            "report_slug": "summary",
            "cart_ref": "local:my-browser-cart",
            "inputs": {},
        })
        self.assertEqual(status, 404, body)
        self.assertIn("local_cart_unsupported", json.dumps(body))

    def test_missing_cart_ref_is_422(self):
        status, body = self._post_generate({
            "report_slug": "summary",
            "inputs": {},
        })
        # Pydantic-level validation error.
        self.assertEqual(status, 422, body)


class TestReportCartsEnumeration(unittest.TestCase):
    """Smoke test for GET /api/reports/carts.

    Materializes a temp cartridges dir with one ``.cart.npz`` (compatible),
    one ``.pkl`` (legacy/incompatible), and one shared-stem entry (both
    formats — should dedupe as compatible). Points ``get_cartridge_dirs``
    at the temp dir via monkeypatching the module attribute the route
    walks, then asserts the response shape, ordering, and dedup rule.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="reports_carts_test_")

        # A compatible cart — a real minimal ``.cart.npz`` (any bytes work
        # for the enumeration step; the endpoint checks existence only).
        cls.compat_path = os.path.join(cls.tmpdir, "aaa-compat.cart.npz")
        np.savez(cls.compat_path, embeddings=np.zeros((1, 4), dtype=np.float32))

        # A legacy .pkl cart.
        cls.legacy_path = os.path.join(cls.tmpdir, "zzz-legacy.pkl")
        with open(cls.legacy_path, "wb") as f:
            f.write(b"legacy-cart-bytes")

        # A stem present in BOTH formats — should be deduped as compatible.
        cls.dual_npz_path = os.path.join(cls.tmpdir, "middle-dual.cart.npz")
        np.savez(cls.dual_npz_path, embeddings=np.zeros((1, 4), dtype=np.float32))
        cls.dual_pkl_path = os.path.join(cls.tmpdir, "middle-dual.pkl")
        with open(cls.dual_pkl_path, "wb") as f:
            f.write(b"legacy-companion")

        # Point the enumeration walk + companion lookup at our tempdir.
        # Both ``get_cartridge_dirs`` (imported into reports_routes) and
        # ``find_companion_file`` (used to double-check the npz resolves)
        # need to be swapped so the answer is deterministic in tests.
        # SANDBOX_DIR gets its own separate empty tempdir so this class's
        # canonical-only fixture doesn't accidentally surface a sandbox
        # entry — the sandbox lookup runs unconditionally now.
        from api import reports_routes as rr
        cls._orig_get_dirs = rr.get_cartridge_dirs
        cls._orig_find_companion = rr.find_companion_file
        cls._orig_sandbox = rr.SANDBOX_DIR
        cls.sandbox_tmpdir = tempfile.mkdtemp(prefix="reports_carts_test_sandbox_")

        def _test_dirs() -> list[str]:
            return [cls.tmpdir]

        def _test_find_companion(name: str, suffix: str) -> Optional[str]:
            path = os.path.join(cls.tmpdir, f"{name}{suffix}")
            return path if os.path.exists(path) else None

        rr.get_cartridge_dirs = _test_dirs  # type: ignore[assignment]
        rr.find_companion_file = _test_find_companion  # type: ignore[assignment]
        rr.SANDBOX_DIR = cls.sandbox_tmpdir  # type: ignore[assignment]

        from fastapi.testclient import TestClient
        from api.main import app
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        from api import reports_routes as rr
        rr.get_cartridge_dirs = cls._orig_get_dirs  # type: ignore[assignment]
        rr.find_companion_file = cls._orig_find_companion  # type: ignore[assignment]
        rr.SANDBOX_DIR = cls._orig_sandbox  # type: ignore[assignment]
        # Best-effort cleanup — leave the tempdir if any file resists.
        for p in (cls.compat_path, cls.legacy_path, cls.dual_npz_path, cls.dual_pkl_path):
            try:
                os.unlink(p)
            except OSError:
                pass
        try:
            os.rmdir(cls.tmpdir)
        except OSError:
            pass
        try:
            os.rmdir(cls.sandbox_tmpdir)
        except OSError:
            pass

    def test_carts_endpoint_shape_and_ordering(self):
        r = self.client.get("/api/reports/carts")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertIn("carts", body)
        carts = body["carts"]

        # Dedupe: 3 unique stems (aaa-compat, middle-dual, zzz-legacy).
        ids = [c["id"] for c in carts]
        self.assertEqual(sorted(set(ids)), sorted(ids), "duplicate cart ids in payload")
        self.assertEqual(set(ids), {"aaa-compat", "middle-dual", "zzz-legacy"})

        # Every entry has the five expected keys (location was added
        # 2026-07-13 with the sandbox walk).
        for c in carts:
            self.assertIn("id", c)
            self.assertIn("display_name", c)
            self.assertIn("report_compatible", c)
            self.assertIn("format", c)
            self.assertIn("location", c)
            self.assertIn(c["format"], ("npz", "pkl"))
            self.assertIn(c["location"], ("canonical", "sandbox"))

        by_id = {c["id"]: c for c in carts}
        self.assertTrue(by_id["aaa-compat"]["report_compatible"])
        self.assertEqual(by_id["aaa-compat"]["format"], "npz")
        self.assertEqual(by_id["aaa-compat"]["location"], "canonical")

        # Dual-format stem dedupes to compatible.
        self.assertTrue(by_id["middle-dual"]["report_compatible"])
        self.assertEqual(by_id["middle-dual"]["format"], "npz")
        self.assertEqual(by_id["middle-dual"]["location"], "canonical")

        # Legacy .pkl only stem is incompatible; display_name marks it.
        self.assertFalse(by_id["zzz-legacy"]["report_compatible"])
        self.assertEqual(by_id["zzz-legacy"]["format"], "pkl")
        self.assertIn("legacy", by_id["zzz-legacy"]["display_name"].lower())
        self.assertEqual(by_id["zzz-legacy"]["location"], "canonical")

        # Ordering: canonical .cart.npz first (alphabetical), sandbox next
        # (this fixture has no sandbox entries), legacy .pkl last. With
        # this fixture: aaa-compat, middle-dual, zzz-legacy.
        self.assertEqual(ids, ["aaa-compat", "middle-dual", "zzz-legacy"])


class TestSandboxCartResolution(unittest.TestCase):
    """End-to-end coverage for sandbox-uploaded cart resolution.

    The Reports-scope resolver walks ``cartridges/_session_uploads/`` in
    addition to the canonical cartridge dirs so that carts uploaded via
    ``POST /api/cartridges/upload`` (which land under a
    ``<12-hex-uuid>_<name>.cart.npz`` filename) are discoverable both by
    ``/api/reports/carts`` and ``/api/reports/generate`` without a further
    mount step.

    Fixture:
      - One canonical ``.cart.npz`` (to keep the ``TestReportCartsEnumeration``
        contract intact for canonical carts).
      - One sandbox ``.cart.npz`` at ``abc123def456_Test.cart.npz``.
      - Enumeration + generate both point at these temp dirs via
        module-attribute monkeypatching.
    """

    STEM = "abc123def456_Test"  # what the frontend dropdown displays as id
    HUMAN = "Test"              # the stripped human-readable form

    CART_PASSAGES = [
        "Delivery invoice from Acme Foods dated 2026-06-01.",
        "Weekly beverage order — total $315.00 on 2026-06-08.",
        "Bar stock replenishment posted 2026-06-15.",
    ]
    CART_SOURCES = [
        "acme-jun01.pdf",
        "beverages-jun08.pdf",
        "bar-stock-jun15.pdf",
    ]

    @classmethod
    def setUpClass(cls) -> None:
        # Two tempdirs: one plays the role of canonical cartridges/, the
        # other the sandbox subdir. Real deployment has the sandbox as a
        # child of cartridges/ but for testing they can be siblings.
        cls.canonical_dir = tempfile.mkdtemp(prefix="reports_sandbox_test_canon_")
        cls.sandbox_dir = tempfile.mkdtemp(prefix="reports_sandbox_test_sandbox_")

        # Materialize the sandbox cart with the full sandbox filename.
        cls.sandbox_cart_path = os.path.join(
            cls.sandbox_dir, f"{cls.STEM}.cart.npz",
        )
        _write_synthetic_cart(
            cls.sandbox_cart_path,
            cls.CART_PASSAGES,
            cls.CART_SOURCES,
            cart_name=cls.HUMAN,
        )

        from api import reports_routes as rr
        cls._orig_get_dirs = rr.get_cartridge_dirs
        cls._orig_find_companion = rr.find_companion_file
        cls._orig_sandbox = rr.SANDBOX_DIR

        def _test_dirs() -> list[str]:
            return [cls.canonical_dir]

        def _test_find_companion(name: str, suffix: str) -> Optional[str]:
            path = os.path.join(cls.canonical_dir, f"{name}{suffix}")
            return path if os.path.exists(path) else None

        rr.get_cartridge_dirs = _test_dirs  # type: ignore[assignment]
        rr.find_companion_file = _test_find_companion  # type: ignore[assignment]
        rr.SANDBOX_DIR = cls.sandbox_dir  # type: ignore[assignment]

        from fastapi.testclient import TestClient
        from api.main import app
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        from api import reports_routes as rr
        rr.get_cartridge_dirs = cls._orig_get_dirs  # type: ignore[assignment]
        rr.find_companion_file = cls._orig_find_companion  # type: ignore[assignment]
        rr.SANDBOX_DIR = cls._orig_sandbox  # type: ignore[assignment]
        for p in (cls.sandbox_cart_path,):
            try:
                os.unlink(p)
            except OSError:
                pass
        for d in (cls.sandbox_dir, cls.canonical_dir):
            try:
                os.rmdir(d)
            except OSError:
                pass

    def _post_generate(self, payload: dict) -> tuple[int, dict]:
        r = self.client.post("/api/reports/generate", json=payload)
        try:
            body = r.json()
        except Exception:
            body = {"_raw": r.text}
        return r.status_code, body

    # -- enumeration -------------------------------------------------------

    def test_sandbox_cart_appears_in_carts_endpoint(self):
        r = self.client.get("/api/reports/carts")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        carts = body.get("carts", [])
        by_id = {c["id"]: c for c in carts}
        self.assertIn(self.STEM, by_id)
        entry = by_id[self.STEM]
        self.assertTrue(entry["report_compatible"])
        self.assertEqual(entry["format"], "npz")
        self.assertEqual(entry["location"], "sandbox")
        # Display name strips the uuid prefix + appends "(sandbox)".
        self.assertEqual(entry["display_name"], f"{self.HUMAN} (sandbox)")

    # -- resolution --------------------------------------------------------

    def test_generate_by_bare_stem_resolves(self):
        # Frontend may send just the human name "Test" — the resolver
        # should suffix-match it to abc123def456_Test.cart.npz.
        status, body = self._post_generate({
            "report_slug": "summary",
            "cart_ref": self.HUMAN,
            "inputs": {"top_themes": 3},
        })
        self.assertEqual(status, 200, body)
        self.assertGreater(len(body["markdown"]), 0)
        # cart_location should surface in metadata for debugging.
        self.assertEqual(body["metadata"].get("cart_location"), "sandbox")

    def test_generate_by_uuid_prefixed_stem_resolves(self):
        # Frontend also uses the full stem "abc123def456_Test" as the
        # cart id (that's what the /carts endpoint returns as id).
        status, body = self._post_generate({
            "report_slug": "summary",
            "cart_ref": self.STEM,
            "inputs": {"top_themes": 3},
        })
        self.assertEqual(status, 200, body)
        self.assertGreater(len(body["markdown"]), 0)
        self.assertEqual(body["metadata"].get("cart_location"), "sandbox")

    def test_ttl_race_returns_410_sandbox_cart_expired(self):
        # Simulate the sandbox cleanup loop evicting the file between
        # enumeration and generate: delete the sandbox file, then post.
        # Since the resolver's own listdir won't see it, the failure
        # actually surfaces as cart_not_found here — the 410
        # sandbox_cart_expired branch fires only for the narrower
        # window where resolve succeeds and open fails. Cover that by
        # temporarily patching the resolver to hand back the deleted
        # path with location=sandbox, then post.
        from api import reports_routes as rr
        orig = rr._resolve_cart_ref
        deleted_path = os.path.join(self.sandbox_dir, "ttl_race_ghost.cart.npz")

        def _ghost_resolver(cart_ref: str) -> "rr.ResolvedCart":
            if cart_ref == "ttl-race-ghost":
                return rr.ResolvedCart(path=deleted_path, location="sandbox")
            return orig(cart_ref)

        rr._resolve_cart_ref = _ghost_resolver  # type: ignore[assignment]
        try:
            status, body = self._post_generate({
                "report_slug": "summary",
                "cart_ref": "ttl-race-ghost",
                "inputs": {},
            })
            self.assertEqual(status, 410, body)
            self.assertIn("sandbox_cart_expired", json.dumps(body))
        finally:
            rr._resolve_cart_ref = orig  # type: ignore[assignment]


def _write_synthetic_cart(
    path: str,
    passages: list[str],
    sources: list[str],
    cart_name: str = "test-cart",
    dim: int = 16,
) -> None:
    """Write a synthetic ``.cart.npz`` at ``path``.

    Same shape as ``_make_synthetic_cart`` above but writes to a
    caller-chosen path (needed so we can drop the fixture into a
    specific sandbox filename with the uuid prefix pattern).
    """
    n = len(passages)
    assert len(sources) == n
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal(size=(n, dim), dtype=np.float32) if n else np.zeros((0, dim), dtype=np.float32)
    if n:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms
    passages_arr = np.array(passages, dtype=object)
    sources_arr = np.array(sources, dtype=object)
    hippo = (
        np.stack([_hippo_row(False) for _ in passages])
        if n else np.zeros((0, 64), dtype=np.uint8)
    )
    pattern0 = np.array(json.dumps({
        "cart_name": cart_name,
        "description": "synthetic sandbox cart for resolver smoke test",
    }), dtype=object)
    np.savez(
        path,
        embeddings=embeddings,
        passages=passages_arr,
        source_paths=sources_arr,
        hippocampus=hippo,
        pattern0=pattern0,
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
