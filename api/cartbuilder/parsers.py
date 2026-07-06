"""
parsers.py — Document text extraction for Cart Builder GUI

File-type coverage expanded 2026-06-14 to match the graphify (safishamsi/graphify, MIT)
file-type matrix Andy flagged 2026-05-11. Approach: use the same upstream Python
libraries graphify uses (BeautifulSoup, python-pptx, yaml, json) rather than
vendoring graphify itself. Code files get raw-text parsing for v1 — AST-aware
chunking via tree-sitter is a v2 improvement once chunking strategy is revisited
across the product suite.

Parser dependencies are imported lazily so that missing optional deps only break
their specific file type, not the whole module.
"""
import os
import re
import json as _json
from pathlib import Path


# Lone UTF-16 surrogates (U+D800–U+DFFF) are an encoding mechanism, not real
# characters. When they appear in Python strings they are decoder artifacts
# (e.g. a buggy UTF-16 path that advanced 2 bytes instead of 4 on an astral
# character, emitting the low surrogate as a stray adjacent "character" right
# after the properly-decoded codepoint). They crash FastAPI's JSON serializer
# at response time. Drop them — astral characters themselves (codepoints
# >= U+10000, like 🅿 U+1F17F) are OUTSIDE this range and are preserved.
_LONE_SURROGATE_RE = re.compile(r'[\ud800-\udfff]')


def scrub_lone_surrogates(text: str) -> str:
    return _LONE_SURROGATE_RE.sub('', text) if text else text


def parse_pdf(filepath: Path) -> list[dict]:
    import fitz  # PyMuPDF
    doc = fitz.open(str(filepath))
    results = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            results.append({"text": text, "page": i + 1, "source": filepath.name})
    doc.close()
    return results


# Day 2 — PDF classification for Image Builder routing. Mirrors the
# frontend classifyPdf() (frontend/src/cart-builder-v2/parsers/pdf.ts).
# Same constants both sides so paired and browser-only builds route the same
# file the same way (design doc Q3 golden-path invariant).
PDF_CLASSIFY_TEXT_THRESHOLD = 500
# Andy 2026-07-05 PM (Grant's pitch deck): expanded from 3 → 15 pages because
# the deck was clean on pages 1-3 and only had broken ToUnicode fonts on
# pages 4-7. Sampling only the head missed the corruption. Cost is
# ~100-500ms extra classify time on a 15-page deck; a 300-page report caps
# out at 15 samples so classify stays fast on long documents.
PDF_CLASSIFY_MAX_PAGES = 15
# Andy 2026-07-05: PDFs with broken font ToUnicode maps return LOTS of
# characters (well over the 500-char threshold) but most are Private Use
# Area / replacement / non-Latin garbage that produces unreadable ingest.
# Two-level check:
#   - Per-page: if ANY sampled page has substantial content (>50 chars)
#     but < PDF_CLASSIFY_PAGE_READABLE_THRESHOLD readable, route to
#     Image Builder. One bad page = broken font used elsewhere in the doc.
#   - Aggregate fallback: if overall readable fraction < threshold, same.
# 0.6 is the initial pick; tune with more samples if false-positives appear.
PDF_CLASSIFY_READABLE_THRESHOLD = 0.6
PDF_CLASSIFY_PAGE_READABLE_THRESHOLD = 0.6
PDF_CLASSIFY_PAGE_MIN_CHARS = 50


def _readable_char_count(text: str) -> int:
    """Count characters that look like real text: printable ASCII, common
    whitespace, or Latin-1/Extended-A/B codepoints (accents, etc). Everything
    else (PUA, replacement, most CJK) counts as unreadable for the purposes
    of ToUnicode-corruption detection."""
    count = 0
    for c in text:
        cp = ord(c)
        if 0x20 <= cp <= 0x7E:
            count += 1
        elif c in "\n\r\t":
            count += 1
        elif 0xA0 <= cp <= 0x24F:
            count += 1
    return count


def classify_pdf(filepath: Path) -> str:
    """Return 'text' if the PDF has extractable readable text, else 'scanned'.

    Sums PyMuPDF's page.get_text() lengths across the first
    PDF_CLASSIFY_MAX_PAGES pages, and independently sums the count of
    readable characters (printable ASCII + Latin extensions). Returns 'text'
    only when total > PDF_CLASSIFY_TEXT_THRESHOLD (500) AND the readable
    fraction >= PDF_CLASSIFY_READABLE_THRESHOLD (0.6). Otherwise routes to
    Image Builder /ocr.

    Failure mode: if PyMuPDF can't open the file, return 'scanned' — Docling
    is more likely to salvage a malformed PDF than PyMuPDF is, and if it
    can't either, the calling code surfaces an OCR failure the user can act
    on (rather than a silent parse-to-empty).
    """
    import fitz
    try:
        doc = fitz.open(str(filepath))
    except Exception:
        return "scanned"
    total_chars = 0
    readable_chars = 0
    corrupt_page_found = False
    try:
        pages_to_check = min(len(doc), PDF_CLASSIFY_MAX_PAGES)
        for i in range(pages_to_check):
            try:
                text = (doc[i].get_text() or "").strip()
                page_len = len(text)
                page_readable = _readable_char_count(text)
                total_chars += page_len
                readable_chars += page_readable
                # Per-page corruption check: substantial content but low
                # readable fraction signals a broken ToUnicode font used
                # somewhere in the document. One bad page taints the whole
                # ingest because the same font likely appears elsewhere.
                if page_len >= PDF_CLASSIFY_PAGE_MIN_CHARS:
                    page_fraction = page_readable / page_len
                    if page_fraction < PDF_CLASSIFY_PAGE_READABLE_THRESHOLD:
                        corrupt_page_found = True
            except Exception:
                # Skip page on error; partial score still useful — a truly
                # scanned PDF stays under threshold no matter which page
                # blows up.
                continue
    finally:
        try:
            doc.close()
        except Exception:
            pass
    readable_fraction = readable_chars / total_chars if total_chars else 0.0
    if total_chars <= PDF_CLASSIFY_TEXT_THRESHOLD:
        return "scanned"
    if corrupt_page_found:
        return "scanned"
    if readable_fraction < PDF_CLASSIFY_READABLE_THRESHOLD:
        return "scanned"
    return "text"


# Image-file extension set. Aligned with image-builder/main.py's
# SUPPORTED_FORMATS list. PDFs are handled via classify_pdf, not here.
_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".heic", ".heif",
    ".tif", ".tiff", ".webp", ".bmp",
}


def is_image_file(filepath: Path) -> bool:
    """True when the file extension is one Image Builder can OCR directly.

    Used by builder.py's per-file routing to decide whether to POST straight
    to Image Builder /ocr (image) vs. classify a PDF first.
    """
    return filepath.suffix.lower() in _IMAGE_EXTENSIONS


def parse_docx(filepath: Path) -> list[dict]:
    import docx
    doc = docx.Document(str(filepath))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if not text.strip():
        return []
    return [{"text": text, "page": None, "source": filepath.name}]


def parse_xlsx(filepath: Path) -> list[dict]:
    import openpyxl
    wb = openpyxl.load_workbook(str(filepath), read_only=True, data_only=True)
    results = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = []
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) if c is not None else "" for c in row]
            line = " | ".join(c for c in cells if c)
            if line.strip():
                rows.append(line)
        if rows:
            text = f"Sheet: {sheet_name}\n" + "\n".join(rows)
            results.append({"text": text, "page": None, "source": f"{filepath.name}:{sheet_name}"})
    wb.close()
    return results


def parse_markdown(filepath: Path) -> list[dict]:
    text = filepath.read_text(encoding="utf-8", errors="replace")
    sections = []
    current = []
    for line in text.split("\n"):
        if line.startswith("## ") and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current))

    results = []
    for i, section in enumerate(sections):
        if section.strip():
            results.append({"text": section.strip(), "page": i + 1, "source": filepath.name})
    return results


def parse_text(filepath: Path) -> list[dict]:
    text = filepath.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return []
    return [{"text": text.strip(), "page": None, "source": filepath.name}]


def parse_rtf(filepath: Path) -> list[dict]:
    from striprtf.striprtf import rtf_to_text
    raw = filepath.read_text(encoding="utf-8", errors="replace")
    text = rtf_to_text(raw)
    if not text.strip():
        return []
    return [{"text": text.strip(), "page": None, "source": filepath.name}]


def parse_html(filepath: Path) -> list[dict]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        # Graceful fallback to raw text if BeautifulSoup not installed
        return parse_text(filepath)
    raw = filepath.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n").strip()
    if not text:
        return []
    return [{"text": text, "page": None, "source": filepath.name}]


def parse_pptx(filepath: Path) -> list[dict]:
    try:
        from pptx import Presentation
    except ImportError:
        return []
    prs = Presentation(str(filepath))
    results = []
    for i, slide in enumerate(prs.slides):
        parts = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text and shape.text.strip():
                parts.append(shape.text.strip())
            if shape.has_table:
                for row in shape.table.rows:
                    line = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                    if line:
                        parts.append(line)
        if parts:
            text = "\n".join(parts)
            results.append({"text": text, "page": i + 1, "source": filepath.name})
    return results


def parse_yaml(filepath: Path) -> list[dict]:
    try:
        import yaml
    except ImportError:
        return parse_text(filepath)
    raw = filepath.read_text(encoding="utf-8", errors="replace")
    try:
        data = yaml.safe_load(raw)
    except Exception:
        return parse_text(filepath)
    text = _flatten_structured(data)
    if not text.strip():
        return []
    return [{"text": text.strip(), "page": None, "source": filepath.name}]


def parse_json(filepath: Path) -> list[dict]:
    raw = filepath.read_text(encoding="utf-8", errors="replace")
    try:
        data = _json.loads(raw)
    except Exception:
        return parse_text(filepath)
    text = _flatten_structured(data)
    if not text.strip():
        return []
    return [{"text": text.strip(), "page": None, "source": filepath.name}]


def parse_jsonl(filepath: Path) -> list[dict]:
    """Newline-delimited JSON. Each line becomes one section so the chunker
    can keep records semantically distinct."""
    results = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = _json.loads(line)
                text = _flatten_structured(obj)
            except Exception:
                text = line
            if text.strip():
                results.append({"text": text.strip(), "page": i, "source": filepath.name})
    return results


def _flatten_structured(data, prefix: str = "") -> str:
    """Flatten a JSON/YAML-style nested structure into readable text.
    Preserves keys as labels so retrieval can match on field names too."""
    lines = []
    if isinstance(data, dict):
        for k, v in data.items():
            key_path = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)):
                nested = _flatten_structured(v, key_path)
                if nested:
                    lines.append(nested)
            else:
                lines.append(f"{key_path}: {v}")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            key_path = f"{prefix}[{i}]" if prefix else f"[{i}]"
            if isinstance(item, (dict, list)):
                nested = _flatten_structured(item, key_path)
                if nested:
                    lines.append(nested)
            else:
                lines.append(f"{key_path}: {item}")
    else:
        lines.append(f"{prefix}: {data}" if prefix else str(data))
    return "\n".join(lines)


# Code-language extensions get raw-text parsing for v1. The 300/50-word
# chunker handles them adequately. v2 will add AST-aware chunking via
# tree-sitter once chunking strategy is revisited across the product suite.
# List mirrors the graphify 5/11 file-type matrix (36+ code languages).
_CODE_EXTENSIONS = {
    # General-purpose
    ".py", ".pyi", ".ipynb",
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".go", ".rs", ".java", ".kt", ".scala",
    ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hh",
    ".rb", ".cs", ".php", ".swift",
    ".lua", ".luau", ".zig",
    # Shell & scripting
    ".sh", ".bash", ".zsh", ".fish",
    ".ps1", ".psm1", ".bat", ".cmd",
    # Functional & ML
    ".ex", ".exs", ".jl", ".elm", ".clj", ".cljs", ".hs",
    # System & DSL
    ".sql", ".graphql", ".gql",
    ".m", ".mm", ".dart",
    # Web frameworks
    ".vue", ".svelte",
    # Build
    ".groovy", ".gradle",
    # Hardware / Verilog
    ".v", ".sv",
    # Fortran
    ".f", ".f90", ".f95", ".f03", ".f08",
    # Pascal / Delphi
    ".pas", ".pp", ".dpr", ".dpk", ".lpr", ".inc", ".dfm", ".lfm", ".lpk",
    # Infra-as-code
    ".tf", ".hcl",
    # GPU / shader
    ".cu", ".cuh", ".wgsl", ".glsl", ".hlsl",
}


def parse_file(filepath: Path) -> list[dict]:
    ext = filepath.suffix.lower()
    parsers = {
        # Existing
        ".pdf": parse_pdf,
        ".docx": parse_docx,
        ".doc": parse_docx,
        ".xlsx": parse_xlsx,
        ".xls": parse_xlsx,
        ".md": parse_markdown,
        ".mdx": parse_markdown,
        ".qmd": parse_markdown,
        ".txt": parse_text,
        ".rtf": parse_rtf,
        # New 2026-06-14 (graphify matrix)
        ".html": parse_html,
        ".htm": parse_html,
        ".pptx": parse_pptx,
        ".ppt": parse_pptx,
        ".yaml": parse_yaml,
        ".yml": parse_yaml,
        ".json": parse_json,
        ".jsonl": parse_jsonl,
        ".ndjson": parse_jsonl,
        ".rst": parse_text,  # raw-text v1; docutils-aware v2
    }
    # Code-extension dispatch
    if ext in _CODE_EXTENSIONS:
        return parse_text(filepath)
    parser = parsers.get(ext)
    if parser:
        return parser(filepath)
    # Fallback: try as text
    return [{"text": filepath.read_text(errors="replace"), "page": None, "source": filepath.name}]


def chunk_texts(sections: list[dict], chunk_size: int = 300, overlap: int = 50) -> list[dict]:
    """Split parsed sections into overlapping word-budgeted chunks.

    Line-aware: chunk boundaries fall between lines, never mid-line, so
    markdown structure (tables, lists, headings, paragraph breaks) survives
    intact and the passage viewer's react-markdown + remark-gfm renders
    them as-intended. Overlap is expressed as trailing lines whose combined
    word count is >= `overlap`.

    Andy 2026-07-05: previous implementation (`text.split()` then
    `" ".join()`) destroyed all newlines, turning Docling OCR output into
    a single wall of piped text that remark-gfm parsed as paragraph, not
    table. This fix restores structural markdown.
    """
    chunks = []
    for section in sections:
        section_text = scrub_lone_surrogates(section["text"])
        words = section_text.split()
        if len(words) <= chunk_size:
            chunks.append({**section, "text": section_text})
            continue
        # Line-aware chunker. Take whole lines only; each chunk fills to
        # roughly `chunk_size` words. Single lines longer than `chunk_size`
        # are taken atomically (rare on real documents; better one oversize
        # chunk than a mangled table row).
        lines = section_text.split("\n")
        line_wc = [len(l.split()) for l in lines]
        n = len(lines)
        i = 0
        part = 0
        while i < n:
            j = i
            budget = 0
            while j < n and (budget == 0 or budget + line_wc[j] <= chunk_size):
                budget += line_wc[j]
                j += 1
            chunk_text = "\n".join(lines[i:j]).strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page": section.get("page"),
                    "source": section["source"],
                    "part": part,
                })
                part += 1
            if j >= n:
                break
            # Walk backward from j to build an `overlap`-word tail that
            # becomes the next chunk's prefix. Guard: always advance at
            # least one line so pathological inputs still terminate.
            back = j
            overlap_wc = 0
            while back > i + 1 and overlap_wc < overlap:
                back -= 1
                overlap_wc += line_wc[back]
            i = back
    return chunks
