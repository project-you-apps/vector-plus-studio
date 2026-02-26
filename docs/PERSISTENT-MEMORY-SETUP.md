# Persistent Session Memory for Claude Code

**Give Claude Code a brain that survives between sessions.**

This guide walks you through setting up persistent memory for Claude Code so that your AI assistant remembers past conversations, learns your codebase over time, and can search its own history semantically. This is the exact setup used by the [Project You](https://project-you.app) team for daily development.

By the end you'll have:

- A **session memory server** that auto-ingests Claude Code transcripts into a searchable cartridge
- An **MCP proxy** so Claude can search and store memories without Bash permission prompts
- A **multi-level memory hierarchy** (L1 curated index, L2 topic files, L3 semantic search, L4 full transcripts)
- Optional: **Membot** for domain-specific brain cartridges and **Membraine** for secure web fetch

## Architecture Overview

```
Claude Code
  |
  |-- L1: MEMORY.md (auto-loaded every session, ~200 lines)
  |       Curated index of key decisions, file paths, conventions.
  |       Claude reads this at the start of every conversation.
  |
  |-- L2: Topic files (memory/*.md)
  |       Detailed notes organized by topic. Claude reads on demand.
  |       Linked from MEMORY.md for fast navigation.
  |
  |-- L3: Session Memory MCP Server (semantic search, ~15ms)
  |       |-- session_mcp_server.py (stdio, launched by Claude Code)
  |       |-- memory_server.py (HTTP, always-on background process)
  |       |-- session_cartridge_builder.py (JSONL parser + embedder)
  |       Thousands of passages from every past session.
  |       Claude searches this when it needs deeper context.
  |
  |-- L4: JSONL transcripts on disk (full fidelity)
  |       Raw Claude Code conversation logs.
  |       Resolved via SOURCE pointers from L3 search results.
  |
  |-- Optional: Membot (domain knowledge cartridges)
  |-- Optional: Membraine (secure web fetch with injection defense)
```

## Prerequisites

- **Python 3.10+**
- **Claude Code** (CLI or VS Code extension)
- ~2 GB RAM for the embedding model (Nomic Embed v1.5, downloads on first run)
- ~270 MB disk for the cached model

## Step 1: Install Dependencies

```bash
pip install sentence-transformers numpy fastapi uvicorn fastmcp einops
```

## Step 2: Create the Session Cartridge Builder

The builder parses Claude Code's JSONL conversation transcripts into searchable passages.

Claude Code stores transcripts at:
```
~/.claude/projects/<project-hash>/*.jsonl
```

Each JSONL file is one conversation session. The builder:
1. Reads all JSONL files in the sessions directory
2. Extracts user messages, assistant text, and thinking blocks
3. Chunks them into passages (~500 chars each)
4. Embeds with Nomic v1.5 (768-dim)
5. Writes a `.pkl` cartridge with embeddings + text + metadata

Create `tools/session_cartridge_builder.py`:

```python
"""
Session cartridge builder — parse Claude Code JSONL transcripts
into an embedded, searchable cartridge.

Usage:
    # Parse only (inspect what you'll get)
    python session_cartridge_builder.py --parse-only --output sessions_parsed.txt

    # Full build (parse + embed + write cartridge)
    python session_cartridge_builder.py --output my_sessions.pkl

    # Custom sessions directory
    python session_cartridge_builder.py --sessions-dir "~/.claude/projects/my-project"
"""

import json
import os
import re
import pickle
import argparse
from pathlib import Path
from datetime import datetime

# Point this at your Claude Code project's sessions directory
DEFAULT_SESSIONS_DIR = os.path.expanduser(
    "~/.claude/projects/<your-project-hash>"
)


class SessionParser:
    """Parse a single Claude Code JSONL session transcript."""

    CONVERSATION_TYPES = {"user", "assistant"}
    TEXT_TYPES = {"text", "thinking"}
    SKIP_CONTENT_TYPES = {"tool_use", "tool_result", "image"}

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.exchanges = []

    def parse(self):
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg_type = obj.get("type")
                if msg_type not in self.CONVERSATION_TYPES:
                    continue

                content = obj.get("message", {}).get("content", "")

                # User messages are often plain strings
                if isinstance(content, str):
                    text = content.strip()
                    if text:
                        self.exchanges.append({
                            "role": msg_type,
                            "text": text,
                            "timestamp": obj.get("timestamp", ""),
                        })
                    continue

                # Assistant messages are arrays of content blocks
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        btype = block.get("type", "")
                        if btype in self.SKIP_CONTENT_TYPES:
                            continue
                        if btype == "text":
                            parts.append(block.get("text", ""))
                        elif btype == "thinking":
                            parts.append(f"[thinking] {block.get('thinking', '')}")
                    text = "\n".join(parts).strip()
                    if text:
                        self.exchanges.append({
                            "role": msg_type,
                            "text": text,
                            "timestamp": obj.get("timestamp", ""),
                        })
        return self.exchanges


def scan_sessions(sessions_dir):
    """Find all JSONL session files."""
    p = Path(sessions_dir)
    if not p.exists():
        print(f"Sessions directory not found: {sessions_dir}")
        return []
    files = sorted(p.glob("*.jsonl"), key=lambda f: f.stat().st_mtime)
    return files


def chunk_text(text, max_chars=500, overlap=50):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def build_passages(sessions_dir):
    """Parse all sessions into chunked passages."""
    files = scan_sessions(sessions_dir)
    passages = []
    for f in files:
        parser = SessionParser(f)
        exchanges = parser.parse()
        session_id = f.stem

        for ex in exchanges:
            role = ex["role"]
            text = ex["text"]

            # Skip very short messages
            if len(text) < 20:
                continue

            # Chunk long messages
            if len(text) > 500:
                chunks = chunk_text(text)
                for i, chunk in enumerate(chunks):
                    passages.append({
                        "text": chunk,
                        "session_id": session_id,
                        "role": role,
                        "chunk": i + 1,
                        "chunks": len(chunks),
                    })
            else:
                passages.append({
                    "text": text,
                    "session_id": session_id,
                    "role": role,
                    "chunk": 1,
                    "chunks": 1,
                })

    return passages


def build_cartridge(passages, output_path):
    """Embed passages and write a searchable cartridge."""
    import numpy as np
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers required.")
        print("  pip install sentence-transformers numpy")
        return

    print(f"Loading Nomic embedding model...")
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    texts = [p["text"] for p in passages]
    # Nomic requires task prefixes
    prefixed = [f"search_document: {t}" for t in texts]

    print(f"Embedding {len(texts)} passages...")
    embeddings = model.encode(prefixed, show_progress_bar=True, convert_to_numpy=True)

    cart = {
        "texts": texts,
        "embeddings": embeddings,
        "metadata": [
            {
                "session_id": p["session_id"],
                "role": p["role"],
                "chunk": p["chunk"],
                "chunks": p["chunks"],
            }
            for p in passages
        ],
    }

    with open(output_path, "wb") as f:
        pickle.dump(cart, f)

    print(f"Wrote {len(texts)} passages to {output_path}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions-dir", default=DEFAULT_SESSIONS_DIR)
    parser.add_argument("--output", default="session_cartridge.pkl")
    parser.add_argument("--parse-only", action="store_true")
    args = parser.parse_args()

    passages = build_passages(args.sessions_dir)
    print(f"Parsed {len(passages)} passages from sessions")

    if args.parse_only:
        for p in passages[:10]:
            print(f"[{p['role']}] {p['text'][:100]}...")
    else:
        build_cartridge(passages, args.output)
```

**Find your project hash**: Look in `~/.claude/projects/` — it's the directory name that corresponds to your project path (with dashes replacing path separators).

## Step 3: Create the Memory Server

The memory server keeps the embedding model warm and the cartridge loaded. Without it, every search would take ~10 seconds to cold-start the model.

Create `tools/memory_server.py`:

```python
"""
Memory server — persistent background search for Claude's session memory.

Keeps Nomic loaded in RAM. Auto-appends new session transcripts.

Usage:
    python tools/memory_server.py
    python tools/memory_server.py --port 8100 --cartridge path/to/cart.pkl

Endpoints:
    GET  /search?q=query&top_k=5  — semantic search
    POST /store?text=...&tags=... — store new passage
    GET  /status                  — health check
    GET  /passage/{idx}           — get passage by index
"""

import os
import pickle
import argparse
import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

# Global state
_model = None
_cart = None
_cart_path = None


def _load_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print("Loading Nomic embedding model...")
        _model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
        )
        print("Model ready.")
    return _model


def _load_cart(path):
    global _cart, _cart_path
    with open(path, "rb") as f:
        _cart = pickle.load(f)
    _cart_path = path
    count = len(_cart.get("texts", []))
    print(f"Loaded cartridge: {count} passages from {path}")


@app.get("/status")
async def status():
    count = len(_cart["texts"]) if _cart else 0
    return {"ready": _cart is not None, "passage_count": count}


@app.get("/search")
async def search(q: str = Query(...), top_k: int = Query(5)):
    import time
    t0 = time.perf_counter()

    model = _load_model()
    query_emb = model.encode(
        [f"search_query: {q}"], convert_to_numpy=True
    )[0]

    embeddings = _cart["embeddings"]
    texts = _cart["texts"]

    # Cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = embeddings / norms
    q_norm = query_emb / (np.linalg.norm(query_emb) or 1)
    scores = normed @ q_norm

    top_idx = np.argsort(scores)[::-1][:top_k]
    elapsed = (time.perf_counter() - t0) * 1000

    results = []
    for rank, idx in enumerate(top_idx, 1):
        results.append({
            "rank": rank,
            "passage_index": int(idx),
            "score": float(scores[idx]),
            "passage": texts[idx],
        })

    return {
        "results": results,
        "total_passages": len(texts),
        "elapsed_ms": round(elapsed, 1),
    }


@app.get("/passage/{idx}")
async def get_passage(idx: int):
    if not _cart or idx < 0 or idx >= len(_cart["texts"]):
        return JSONResponse({"error": "Invalid index"}, 404)
    return {"passage": _cart["texts"][idx], "index": idx}


@app.post("/store")
async def store(text: str = Query(...), tags: str = Query("")):
    model = _load_model()
    emb = model.encode(
        [f"search_document: {text}"], convert_to_numpy=True
    )[0]

    _cart["texts"].append(text)
    _cart["embeddings"] = np.vstack([_cart["embeddings"], emb.reshape(1, -1)])

    if "metadata" in _cart:
        _cart["metadata"].append({"tags": tags.split(",") if tags else []})

    # Persist immediately
    with open(_cart_path, "wb") as f:
        pickle.dump(_cart, f)

    return {
        "stored": True,
        "passage_index": len(_cart["texts"]) - 1,
        "total_passages": len(_cart["texts"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--cartridge", default="session_cartridge.pkl")
    args = parser.parse_args()

    _load_model()
    _load_cart(args.cartridge)

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
```

### First build + start

```bash
# 1. Build the initial cartridge from your session transcripts
python tools/session_cartridge_builder.py \
    --sessions-dir ~/.claude/projects/<your-project-hash> \
    --output session_cartridge.pkl

# 2. Start the memory server (keep this running)
python tools/memory_server.py --cartridge session_cartridge.pkl
```

Test it:
```bash
curl "http://localhost:8100/status"
curl "http://localhost:8100/search?q=how+does+authentication+work&top_k=3"
```

## Step 4: Create the Session MCP Server

This is the bridge that gives Claude Code direct tool access to your memory server. It runs as a stdio MCP server — Claude Code launches it automatically.

Create `tools/session_mcp_server.py`:

```python
"""
Session MCP server — gives Claude Code tools to search/store session memory.
Proxies to the HTTP memory server on localhost:8100.
"""

from fastmcp import FastMCP
import urllib.request
import urllib.parse
import json
import sys
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stderr)
log = logging.getLogger("session-mcp")

MEMORY_URL = "http://localhost:8100"
TIMEOUT = 15

mcp = FastMCP("SessionMemory")

NOT_RUNNING = "Session memory server not running. Start it with: python tools/memory_server.py"


def _get(path, params=None):
    url = f"{MEMORY_URL}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    resp = urllib.request.urlopen(url, timeout=TIMEOUT)
    return json.loads(resp.read().decode("utf-8"))


def _post(path, params=None):
    url = f"{MEMORY_URL}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, method="POST", data=b"")
    resp = urllib.request.urlopen(req, timeout=TIMEOUT)
    return json.loads(resp.read().decode("utf-8"))


@mcp.tool()
def session_search(query: str, top_k: int = 5) -> str:
    """Search the session memory cartridge using semantic similarity.
    Returns ranked results from all past Claude Code sessions.

    Args:
        query: Natural language search query
        top_k: Number of results to return (default 5)
    """
    try:
        data = _get("/search", {"q": query, "top_k": top_k})
    except urllib.error.URLError:
        return NOT_RUNNING

    results = data.get("results", [])
    if not results:
        return f"No results for '{query}'"

    lines = [f"Found {len(results)} results ({data.get('total_passages', 0)} passages):\n"]
    for r in results:
        lines.append(f"#{r['rank']} [{r['score']:.3f}] passage #{r['passage_index']}")
        passage = r.get("passage", "")
        lines.append(passage[:600] + "..." if len(passage) > 600 else passage)
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
def session_store(content: str, tags: str = "") -> str:
    """Store new text in the session memory cartridge.
    Embedded with Nomic and persisted to disk immediately.

    Args:
        content: Text content to store (max 10,000 chars)
        tags: Optional comma-separated tags (e.g. 'ARCHITECTURE,DECISION')
    """
    try:
        data = _post("/store", {"text": content, "tags": tags} if tags else {"text": content})
    except urllib.error.URLError:
        return NOT_RUNNING

    if data.get("stored"):
        return f"Stored as passage #{data['passage_index']} ({data['total_passages']} total)"
    return f"Store failed: {data}"


@mcp.tool()
def session_status() -> str:
    """Get session memory server status: passage count, readiness."""
    try:
        data = _get("/status")
    except urllib.error.URLError:
        return NOT_RUNNING
    return f"Session Memory: {'READY' if data['ready'] else 'NOT READY'}, {data['passage_count']} passages"


if __name__ == "__main__":
    log.info("Starting Session Memory MCP Server (stdio)")
    mcp.run(transport="stdio")
```

## Step 5: Register MCP Servers with Claude Code

Add the session memory server to your Claude Code config. Edit `~/.claude.json` and add to the `mcpServers` section:

```json
{
  "mcpServers": {
    "session-memory": {
      "command": "python",
      "args": ["/absolute/path/to/tools/session_mcp_server.py"]
    }
  }
}
```

> **Important**: Use the absolute path to `session_mcp_server.py`. Claude Code launches MCP servers as subprocesses from its own working directory, so relative paths won't work.

After restarting Claude Code, you'll see the new tools available:

| Tool | Description |
|------|-------------|
| `session_search` | Semantic search across all past sessions |
| `session_store` | Store a new passage (insight, decision, journal entry) |
| `session_status` | Check if the memory server is running and how many passages are loaded |

Claude can now search its own history:

```
> Search session memory for "how we implemented authentication"
```

## Step 6: Set Up Auto-Memory (MEMORY.md)

Claude Code has a built-in auto-memory feature. When enabled, it creates a `MEMORY.md` file in your project's memory directory:

```
~/.claude/projects/<project-hash>/memory/MEMORY.md
```

This file is **automatically loaded into every conversation context**. Use it as a curated L1 index — a table of contents for everything Claude needs to know at session start.

### Structure

Keep `MEMORY.md` under 200 lines. Use it as an index that links to detailed topic files:

```markdown
# Project Memory

## Architecture
- Backend: FastAPI + PostgreSQL
- Frontend: React 19 + TypeScript + Tailwind
- See [architecture.md](architecture.md) for full details

## Conventions
- Always use snake_case for Python, camelCase for TypeScript
- Tests go in __tests__/ directories adjacent to source
- See [conventions.md](conventions.md) for the full style guide

## Key Decisions
- Chose JWT over sessions for auth (2026-01-15)
- Migrated from SQLite to PostgreSQL (2026-02-01)
- See [decisions.md](decisions.md) for rationale

## Critical File Paths
- API entry: src/api/main.py
- Auth middleware: src/api/auth.py
- Database models: src/models/
```

### Topic files (L2)

Create separate `.md` files in the same memory directory for detailed notes:

```
memory/
├── MEMORY.md              # L1 index (auto-loaded, ~200 lines)
├── architecture.md        # L2 detail
├── conventions.md         # L2 detail
├── decisions.md           # L2 detail
├── debugging.md           # L2 detail
└── deployment.md          # L2 detail
```

Claude reads these on demand when it needs deeper context than what's in MEMORY.md.

## Step 7: Create a Startup Script

Automate starting the memory server alongside your other development services.

### Windows (startup.bat)

```batch
@echo off
echo Starting Memory Server (port 8100)...
start "Memory Server" cmd /k "python tools\memory_server.py --cartridge session_cartridge.pkl"
timeout /t 3 /nobreak >nul
echo Memory server started.
```

### Linux/macOS (startup.sh)

```bash
#!/bin/bash
echo "Starting Memory Server (port 8100)..."
python tools/memory_server.py --cartridge session_cartridge.pkl &
sleep 3
echo "Memory server started. PID: $!"
```

### systemd (Linux, persistent)

```ini
[Unit]
Description=Claude Session Memory Server
After=network.target

[Service]
ExecStart=/path/to/venv/bin/python /path/to/tools/memory_server.py --cartridge /path/to/session_cartridge.pkl
Restart=always
WorkingDirectory=/path/to/project

[Install]
WantedBy=multi-user.target
```

## Optional: Add Membot for Domain Knowledge

[Membot](https://github.com/project-you-apps/membot) is an MCP server for domain-specific brain cartridges. While session memory stores *your conversation history*, Membot stores *reference knowledge* — documentation, research papers, wiki articles, anything you want Claude to be able to search.

### Quick setup

```bash
git clone https://github.com/project-you-apps/membot.git
cd membot
pip install -r requirements.txt

# Build a cartridge from your docs
python cartridge_builder.py ./my-docs/ --name my-knowledge

# Start the server
python membot_server.py --transport http --port 8200 --writable
```

### Register with Claude Code

For a local server, add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "membot": {
      "command": "python",
      "args": ["/path/to/membot/membot_server.py"]
    }
  }
}
```

For a remote server:

```json
{
  "mcpServers": {
    "membot": {
      "type": "http",
      "url": "http://your-server:8200/mcp/",
      "headers": {
        "Authorization": "Bearer your-key-here"
      }
    }
  }
}
```

## Optional: Add Membraine for Secure Web Fetch

[Membraine](https://github.com/project-you-apps/membraine) gives Claude Code a secure web fetch tool with prompt injection defense. Instead of raw HTML, pages go through a 5-layer pipeline: render, extract, convert, scan for poison text, chunk and embed.

### Quick setup

```bash
git clone https://github.com/project-you-apps/membraine.git
cd membraine
pip install -r requirements.txt
playwright install chromium
```

### Register per-project

Add to your project's `.mcp.json` (in the project root):

```json
{
  "mcpServers": {
    "membraine": {
      "type": "stdio",
      "command": "python",
      "args": ["/path/to/membraine/membraine_server.py"]
    }
  }
}
```

## The Memory Hierarchy in Practice

Here's how the four levels work together during a typical Claude Code session:

### Session start
1. Claude reads **MEMORY.md** (L1) — gets the lay of the land in ~200 lines
2. If the task involves a specific subsystem, Claude reads the relevant **topic file** (L2)

### During work
3. Claude **searches session memory** (L3) when it needs context from a past conversation:
   - "How did we implement the auth middleware last week?"
   - "What was the root cause of that deployment bug?"
   - "What encoding did we settle on for the binary format?"
4. For full transcript fidelity, Claude uses **deep search** (L4) to resolve SOURCE references back to the original JSONL

### At session end
5. Claude **stores key decisions and insights** into session memory for future sessions
6. Claude **updates MEMORY.md** if a significant architectural decision was made

### Between sessions
7. The memory server **auto-appends** new session transcripts to the cartridge
8. The cartridge grows over time — our production cartridge has 12,000+ passages spanning 3 months of daily use

## Troubleshooting

**"Session memory server not running"**
The MCP proxy can't reach `localhost:8100`. Start the memory server:
```bash
python tools/memory_server.py --cartridge session_cartridge.pkl
```

**"No results" for everything**
The cartridge might be empty. Build it first:
```bash
python tools/session_cartridge_builder.py --output session_cartridge.pkl
```

**MCP tools not showing up in Claude Code**
- Check `~/.claude.json` has the correct absolute path
- Restart Claude Code after config changes
- Check stderr output: `python tools/session_mcp_server.py` should print "Starting Session Memory MCP Server"

**Search is slow (~10 seconds)**
The embedding model isn't loaded yet. The memory server keeps it warm — make sure it's running before you start Claude Code.

**Cartridge is too large**
The `.pkl` grows as passages accumulate. For a project with 3 months of daily use, expect ~50-100 MB. This is fine for search performance (cosine similarity is fast even at 10K+ passages). If you need to trim, rebuild with `--sessions-dir` pointed at recent sessions only.

## File Summary

```
your-project/
├── tools/
│   ├── session_cartridge_builder.py  # JSONL parser + embedder
│   ├── memory_server.py              # Always-on search server (port 8100)
│   └── session_mcp_server.py         # MCP proxy (stdio, launched by Claude Code)
├── session_cartridge.pkl              # Your brain cartridge (auto-grows)
├── startup.bat                        # Start everything
└── .mcp.json                          # Per-project MCP config (Membraine, etc.)

~/.claude.json                         # User-level MCP config (session-memory, membot)
~/.claude/projects/<hash>/memory/
├── MEMORY.md                          # L1 curated index (auto-loaded)
├── architecture.md                    # L2 topic files
├── decisions.md
└── ...
```

## Links

- [Membot](https://github.com/project-you-apps/membot) — Brain cartridge server for AI agents
- [Membraine](https://github.com/project-you-apps/membraine) — Secure web fetch with injection defense
- [Vector+ Studio](https://github.com/project-you-apps/vector-plus-studio) — GUI for building and searching brain cartridges
- [Project You](https://project-you.app) — Parent project

---

*Your AI shouldn't start every conversation with amnesia.*
