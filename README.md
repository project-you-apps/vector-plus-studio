# Vector+ Studio

**Drag documents in. Search them. Write down why they matter — and keep that reasoning attached, attributed, and correctable.**

Vector+ Studio is the browser front end for a portable document store. Drop in PDFs,
spreadsheets, screenshots or notes and it builds you a **cartridge**--one file holding the
passages, their embeddings, and a compact signature per passage. Search it, annotate it, share
it, or hand the file to someone else.

**Try it:** **https://project-you.app/vps/app**: sample carts are public and searchable without
signing up. Sign in if you have a private library.

⭐ The same cartridges are readable by agents through
**[Membot](https://github.com/project-you-apps/membot)**, over standard MCP. One store, two
front ends: people here, agents there, **the same records and the same notes.**

---

## What it is for

Documents survive. The reasoning behind them does not.

Six months after a decision, the file is still in the folder and the person who chose it has
moved on. Vector+ exists so the *why* is attached to the passage, carries a name and a date, and
is still there when someone goes looking.

**Three properties it is built around:**

**Retrieval costs no tokens.** Searching calls no language model. Embeddings come from a
sentence transformer; ranking is binary matching plus keyword reranking. ⇒ Search as much as you
like, the cost is latency, not spend.

**Every write is attributed.** Who, which surface, and when, on every passage and every note. Not
a log beside the data: a property of the format.

**Corrections never erase.** Supersede a note and the old one stays, struck through, still
attributed. ⇒ A reader can see that someone was wrong, who said so, and when they said it.

---

## The screens

| | |
| --- | --- |
| **Search** | Search and edit the mounted cart. Results carry their source, their notes, and who wrote them |
| **Overview** | Cart stats, what is mounted, system health |
| **Cart Builder** | Drag and drop documents; the browser builds a cart while you watch |
| **Edit Carts** | Add, update and delete passages on the mounted cart |
| **Reports** | Structured views — Summary, Timeline, Trend, Comparison, Entity Rollup, Change Log, Executive TL;DR |
| **Agents** | Scoped recipes — Auto-Briefing, Q&A, Professor, Curator — run against the cart |
| **Profile** | Your identity, your seat, and every cart you can reach |
| **Settings** | Search modes, theme, advanced options |

ⓘ **SQL** is in the nav and disabled because the interpreter is not built yet. It is visible so you
know it is coming, not because it works.

---

## What goes in

Built in the browser, no install:

**PDF** (text and scanned, with table extraction) · **Word** · **PowerPoint** · **Excel** ·
**Markdown** · **HTML** · **RTF** · **JSON / JSONL / YAML** · **plain text and source files**

With the optional desktop companions:

**Screenshots and images**: `png` `jpg` `jpeg` `tiff` `webp` `bmp` `heic`, OCR'd locally
**Large builds**: pair the
[Desktop Builder](https://github.com/project-you-apps/vector-plus-desktop-builder) and the web UI
delegates automatically. Your documents never leave your machine.

---

## Notes — the part people actually ask for

Any passage can carry notes. A note has a **kind** (why, context, caution, decision,
prediction, recommendation) plus an author, a surface and a timestamp.

- **Notes live beside the cart**, in a sidecar file. Annotating never rewrites your data.
- **Correcting supersedes.** The old note remains, struck through and attributed.
- **A hash-chained journal** records every note event, each line carrying the previous line's
  hash, so an edited, reordered or interior-deleted entry is detectable.
- **Notes are private until shared.** Write freely; publish deliberately.

⭐ **Worked example.** A finance cart holds a 2026 expense policy. A controller adds:
*"we set the threshold at $10,000 because the pilot geography was still open"*
The reasoning that exists in nobody's inbox. Three weeks later she corrects herself.
⇒ Both notes are in the record, hers, dated, the first struck through. **Nobody rewrote anything.**

---

## The Hot Stack — available through Membot, not yet in this UI

A shared cart does not have *a* hot stack. Two people searching the same store build different
hierarchies over it, and both are right. So attention is kept **per seat** and rolled up
separately:

- **`seat_hot_stack`**: what *you* have been working on in the mounted cart, scored only over
  passages you actually touched. Another seat's forty reads leave it cold for you.
- **`firm_hot_stack`**: what the **organization** is warm on, which is a different question.
  ⭐ The signal that exists only here is **how many distinct people touched a passage**. One
  person reading something forty times and forty people reading it once produce the same access
  count and mean completely different things about a company.
- **`seat_hide`**: drop something from your own view without removing it for anyone else.

⛔ **These are [Membot](https://github.com/project-you-apps/membot) tools, reachable by agents
over MCP. Studio does not surface them yet.**
There is no Hot Stack screen in the nav, and we would rather say so than imply one.
⬜ A Studio view is on the roadmap.

ⓘ **What it is not.** It is attention over a mounted cart, not a personal memory that follows
you between carts. **Promotion** (moving session work into a durable personal store) and an
auto-mounted personal cart on sign-in are designed and not built yet.

---

## Access

Sign-in is through Supabase. Each signed-in person is a **seat**, and a seat has a level on each
cart it can reach:

| level | may |
| --- | --- |
| `viewer` | read |
| `commenter` | read, annotate |
| `editor` | read, annotate, write |

Per-document control sits underneath that: individual source documents inside a shared cart can
be withheld from a seat, and a withheld document does not surface in that seat's search results
at all.

⚠ **Honest about where the boundary is.** Access control binds when the API is the only way in:
a hosted or intranet deployment. ⛔ On a local install the person running it owns the disk, and
the filesystem outranks the application. ⇒ Local mode is about **data residency**: your documents
stay on your machine. Shared deployment is where the boundary does real work.

---

## Running it yourself

### Prerequisites

Python 3.10+ · Node 20+ · no GPU required

### Install and run

```bash
git clone https://github.com/project-you-apps/vector-plus-studio.git vector-plus-studio-repo
cd vector-plus-studio-repo
pip install -r requirements.txt
cd frontend && npm install && cd ..

# backend on :8000
python run_api.py --host 127.0.0.1 --port 8000

# frontend on :5173, in a second terminal
cd frontend && npm run dev
```

Then open **http://localhost:5173**.

⚠ `frontend/.env.local` needs `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY`. Without them the
app throws at load and the page renders blank with no error on screen.

ⓘ **First search downloads the embedding model** (~1.1 GB, once). Do it on a connection you like.

ⓘ Cartridges are read from `cartridges/` beside the repo. `VPS_EXTRA_CARTRIDGE_DIRS` adds more,
separated by `;` on Windows and `:` elsewhere.

### Where it runs

| | |
| --- | --- |
| **Local** | Your machine. Documents never leave it |
| **Intranet** | One server, many seats, access control binding, nothing facing the internet |
| **Hosted** | `project-you.app/vps/app`, or your own deployment |

---

## Honest status

| | |
| --- | --- |
| Cart building, search, reports, agents | ✅ shipped |
| Notes, correction-by-supersession, journal | ✅ shipped |
| Seats, per-cart levels, per-document withholding | ✅ shipped |
| OCR and spreadsheet ingestion | ✅ shipped via the desktop companions |
| Hot Stack, per seat and firm-wide | ✅ shipped in Membot — ⬜ no Studio view yet |
| SQL interpreter | ⬜ not built — visible in the nav, disabled |
| Sealed carts — a one-way, append-only posture | ⬜ designed |
| Change notification to subscribers | ⬜ designed |

ⓘ **On "associative" search.** Studio can surface neighbors a literal query would not have
found. Having tested it, we attribute that to the **embedding model's** representation quality
rather than to anything downstream of it. The repository includes an optional GPU engine under a
separate licence; it is **not required**, and it is not what makes search work.

---

## Related

- **[Membot](https://github.com/project-you-apps/membot)**: the MCP server. Same carts, read by
  agents.
- **[AGENTS.md](AGENTS.md)**: orientation for an agent working on this codebase.
- **[Desktop Builder](https://github.com/project-you-apps/vector-plus-desktop-builder)**: local
  builds and OCR.

## Licence

| Component | Licence |
| --- | --- |
| Application, API and utilities | MIT |
| Optional GPU engine (`bin/*`) | Proprietary — see [bin/LICENSE](bin/LICENSE) |

The GPU engine is free for personal, educational and non-commercial use. Commercial use requires
a separate licence--[contact us](mailto:andy@project-you.app).
