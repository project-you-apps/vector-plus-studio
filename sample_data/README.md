# Sample Data — Pre-built Carts + Source-Type Test Vaults

This directory contains two kinds of sample assets:

1. **Pre-built `.pkl` cartridges** — ready to mount and search via the
   existing curated-cart workflow. No build pipeline needed.
2. **Source-type test vaults** — folders of raw documents you can drag
   into the in-browser Cart Builder to exercise specific adapters
   (Obsidian, Notion). These exercise the *build* pipeline end-to-end,
   then produce a `.cart.npz` that mounts via the same workflow.

---

## Pre-built cartridges (`.pkl`)

| File | Patterns | Domain | Suggested queries |
| --- | --- | --- | --- |
| `Linux Manual Pages.pkl` | ~1k | Linux man-page corpus | "how do I change file permissions", "process listing utilities" |
| `Python Style Guide.pkl` | small | PEP 8 and style guidance | "naming conventions", "indentation rules" |
| `nomic_dataset_10k.pkl` | 10k | Nomic Atlas sample | open-ended exploratory queries |
| `wiki_nomic_5k.pkl` | 5k | Wikipedia sample | "Who developed quantum theory?" |
| `wiki_nomic_10k.pkl` | 10k | Wikipedia sample | broader queries |
| `wiki_nomic_25k.pkl` | 25k | Wikipedia sample | broader still |
| `wiki_nomic_100k.pkl` | 100k | Wikipedia sample | **Use this for the Lou Costello demo:** mount it, search `Who's on first?` — Lou Costello surfaces R@1 on Associate mode (substrate inference) while cosine modes land on Benjamin Franklin (semantic "first famous person" anchor). Same shape as the README's Poseidon/earthquakes example. |

**To mount:** drag any `.pkl` file from this folder into the Cart
Builder's mount panel, or use the file picker. Read-only by default —
the curated carts ship as immutable demo content.

---

## Source-type test vaults

These folders exercise the in-browser Cart Builder's parser registry
end-to-end. Each vault demonstrates how a different source-type
adapter normalizes its native conventions into the unified cart format.

### `obsidian-vault/` — Obsidian markdown convention

8 markdown files arranged in `People/`, `Concepts/`, and `Daily Notes/`
subdirectories. Renaissance astronomy theme (Galileo / Kepler / Tycho /
heliocentrism / telescope) — public-domain content, identical narrative
to the `notion-export/` sibling for direct adapter comparison.

**Exercises:**
- YAML frontmatter pretty-printing (`title:`, `tags:`, `aliases:`, `born:`, `died:`)
- Wikilinks `[[Note]]` and aliased `[[Note|display text]]`
- Nested wikilinks `[[Note#Heading]]`
- Tags inline `#tag` and nested `#tag/subtag`
- Multi-tag lines, block IDs `^abc`, blockquotes, code blocks
- Folder structure preserved via `webkitRelativePath` (source field
  reads `People/Galileo Galilei.md`, not just `Galileo Galilei.md`)

**To test:** open Cart Builder, drag the `obsidian-vault/` folder in
(directory drop) or multi-select all 8 `.md` files. Build → download
`.cart.npz` → mount → search.

**Suggested queries:**
- *"Who proposed elliptical orbits?"* → should surface Johannes Kepler R@1
- *"telescope inventor"* → Galileo's telescope section + Kepler's optics
- *"naked-eye precision"* → Tycho's observational precision section
- *"Why did heliocentrism take so long?"* → Heliocentrism's "Why it took so long" section

### `notion-export/` — Notion export convention

4 markdown files arranged in `People/` and `Concepts/` subdirectories,
with Notion's URL-encoded + 32-character-hash filename pattern (a page
titled "Galileo Galilei" exports as
`Galileo Galilei 26b5e1f0aef4435faaab0987654321ef.md`). Same Renaissance
content as the Obsidian vault for adapter parity.

**Exercises:**
- Filename hash-stripping (32-char hex suffix removed from source field)
- URL-decoded spaces (`Project%20You...md` → `Project You.md`)
- Inline `Property: value` lines (Notion's properties — no YAML frontmatter)
- Standard markdown body (Notion-flavored markdown == standard markdown
  for our purposes)
- Folder structure preserved

**To test:** same workflow as Obsidian. After build + mount, check that
result-card source fields read cleanly (e.g., `People/Galileo Galilei.md`
not `People/Galileo Galilei 26b5e1f0aef4435faaab0987654321ef.md`).

**Suggested queries:** same as Obsidian set above — substrate retrieval
should produce comparable results since the content is intentionally
parallel.

---

## QA workflow for new source-type adapters

When extending the parser registry with a new source-type:

1. Create a synthetic vault under `sample_data/<source-type>/` using
   public-safe content (Renaissance astronomy keeps comparability with
   existing samples)
2. Run the parser unit-test pattern (see commit history for Obsidian
   and Notion examples — Node harness exercising the helper functions)
3. End-to-end: drag-drop into Cart Builder, build, download, mount,
   search using the example queries
4. Update this README's source-type section with what the adapter
   normalizes and what example queries surface
5. Update the existing-adapter table at the top if your new adapter
   handles a previously-fallback file type

---

## Why pre-built `.pkl` carts and not pre-built `.cart.npz`?

The `.pkl` format is the desktop / V1.0 Streamlit-era format and is
what the curated-cart workflow currently expects. Browser-built carts
emit `.cart.npz` (modular NPZ container with manifest + hippocampus +
permissions sidecars). Both formats mount via the same backend; the
NPZ format is what users build via Cart Builder. Sample folders are
NOT pre-built `.cart.npz` because the build itself is the exercise —
the point of `obsidian-vault/` is to test the parse → chunk → embed →
write pipeline end-to-end.
