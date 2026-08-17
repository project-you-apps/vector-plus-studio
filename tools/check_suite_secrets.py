"""Refuse to package vps-suite if it carries a credential.

WHY THIS EXISTS. `vps-suite` is the DISTRIBUTION UNIT -- it goes on a portable drive, into
Google Drive, and is served from the droplet at /vps/app/downloads/. Anything inside it is
effectively published. We agreed to write this check on 2026-07-23 and did not, and on
2026-08-17 the suite was found shipping `report-builder/.env` with a populated 64-character
`WORKER_AUTH_TOKEN` -- the shared secret with our Cloudflare Worker, which is spend on our
account. It went to Google Drive and to a laptop.

⚠ IT WAS NOT CAUGHT BY EYE, AND IT WOULD NOT HAVE BEEN. The suite is ~1.5 GB across seven
component directories that are refreshed by hand from six different source trees. "Look before
you zip" is not a control at that size; a script that exits non-zero is.

WHAT IT LOOKS FOR
  1. Any file named `.env` / `*.env` -- config that is meant to be per-install, never shipped.
  2. JWTs whose payload carries a `role` claim (Supabase anon / service_role).
  3. High-entropy assignments to names that look like secrets, populated with a real value.
  4. Private key blocks and common provider token prefixes.

It deliberately does NOT try to be clever about which secrets are "safe". A publishable key is
fine to ship and will be reported anyway -- reviewing three known-good hits costs seconds; the
one unknown hit is the whole point.

Usage:
    python tools/check_suite_secrets.py d:/vps-suite
    python tools/check_suite_secrets.py d:/vps-suite --allow-publishable

Exit codes: 0 clean, 1 findings, 2 bad invocation.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import sys

SKIP_DIRS = {"node_modules", "__pycache__", ".git", "venv", ".venv", "models", "dist-ssr"}
MAX_BYTES = 8_000_000

JWT_RE = re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.([A-Za-z0-9_-]{20,})\.[A-Za-z0-9_-]{10,}")
# NAME=VALUE where the name smells like a secret and the value is substantial.
#
# ⚠ TWO PATTERNS, NOT ONE, AND THE REASON MATTERS. The first version applied a single greedy
# rule everywhere and returned 187 findings, ~185 of them `token = request.headers.get(...)`
# in ordinary Python. A gate that cries wolf 187 times is worse than no gate: it gets muted,
# and then it is not a gate. So source code is held to a much stricter rule than config.
#
# CONFIG (.env/.ini/.json/.yaml/.bat/...): a bare NAME=value is exactly how a secret is
# written, so a substantial value is enough to flag.
# ⚠ NOT anchored to line start. The first version was (`^[\s#]*NAME=`), which reads .env
# correctly and MISSES JSON and YAML entirely -- `{"WORKER_AUTH_TOKEN": "..."}` has the key
# mid-line behind a brace. Caught by the decoy test, not by reading the regex.
CONFIG_ASSIGN_RE = re.compile(
    r"(?i)[\"']?\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|PRIVATE_?KEY|CREDENTIAL)"
    r"[A-Z0-9_]*)[\"']?\s*[:=]\s*[\"']?([^\s\"';,#}]{16,})")

# SOURCE (.py/.ts/.js/...): only a QUOTED LITERAL counts. `token = get_header()` is code;
# `TOKEN = "9f3c...64 hex chars"` is a leak. Entropy is checked on top of this.
SOURCE_ASSIGN_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|PRIVATE_?KEY|CREDENTIAL)[A-Z0-9_]*)"
    r"\s*[:=]\s*[\"']([^\"'\s]{16,})[\"']")

CONFIG_EXT = {".env", ".ini", ".cfg", ".conf", ".json", ".yaml", ".yml", ".toml",
              ".bat", ".cmd", ".ps1", ".sh", ".properties"}


def shannon(value: str) -> float:
    """Bits per character. A real token is near-random; an identifier or URL is not."""
    import math
    from collections import Counter
    if not value:
        return 0.0
    n = len(value)
    return -sum((c / n) * math.log2(c / n) for c in Counter(value).values())
PROVIDER_RE = re.compile(r"\b(sk-ant-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|"
                         r"github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{32,})")
PEM_RE = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----")

# Values that are obviously not real. A packager writing a template should not trip the gate.
PLACEHOLDER = re.compile(r"(?i)^(x{4,}|your[-_ ]|<.*>|changeme|placeholder|todo|example|"
                         r"\$\{.*\}|%.*%|\.\.\.)")


def looks_placeholder(value: str) -> bool:
    return bool(PLACEHOLDER.match(value)) or len(set(value)) <= 4


def scan(root: str, allow_publishable: bool) -> list[tuple[str, str, str]]:
    findings: list[tuple[str, str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root)

            if name == ".env" or name.endswith(".env"):
                findings.append((rel, "env-file", "config file, never ship"))
                continue

            try:
                if os.path.getsize(path) > MAX_BYTES:
                    continue
                text = io.open(path, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue

            if PEM_RE.search(text):
                findings.append((rel, "private-key", "PEM private key block"))

            for payload in set(JWT_RE.findall(text)):
                try:
                    claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
                except Exception:                                   # noqa: BLE001
                    continue
                role = claims.get("role")
                if not role:
                    continue
                if role == "anon" and allow_publishable:
                    continue
                findings.append((rel, f"jwt:{role}", "Supabase key embedded in a shipped file"))

            for m in PROVIDER_RE.finditer(text):
                findings.append((rel, "provider-token", m.group(1)[:8] + "..."))

            is_config = os.path.splitext(name)[1].lower() in CONFIG_EXT
            pattern = CONFIG_ASSIGN_RE if is_config else SOURCE_ASSIGN_RE
            for var, value in pattern.findall(text):
                if looks_placeholder(value):
                    continue
                # Entropy floor. `WORKER_AUTH_TOKEN=<64 hex>` scores ~4; a path, a URL, an
                # import name or an English phrase scores well under 3.5. Config is held to a
                # lower bar than source because a config value is *expected* to be the secret.
                floor = 3.0 if is_config else 3.6
                if shannon(value) < floor:
                    continue
                findings.append((rel, "assignment",
                                 f"{var}= ({len(value)} chars, entropy {shannon(value):.1f})"))
    return findings


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    if len(args) != 1 or not os.path.isdir(args[0]):
        print(__doc__.strip().splitlines()[-4], file=sys.stderr)
        print("usage: check_suite_secrets.py <dir> [--allow-publishable]", file=sys.stderr)
        return 2
    root = args[0]
    allow_pub = "--allow-publishable" in argv

    findings = scan(root, allow_pub)
    if not findings:
        print(f"CLEAN — no credentials found under {root}")
        return 0

    print(f"REFUSING TO PACKAGE — {len(findings)} finding(s) under {root}:\n")
    width = max(len(f[0]) for f in findings)
    for rel, kind, detail in sorted(findings):
        print(f"  {rel:<{width}}  {kind:<16} {detail}")
    print("\nMove real secrets OUT of the suite and ship a .env.example instead.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
