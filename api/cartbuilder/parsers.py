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
    """Split parsed sections into overlapping word-based chunks."""
    chunks = []
    for section in sections:
        section_text = scrub_lone_surrogates(section["text"])
        words = section_text.split()
        if len(words) <= chunk_size:
            chunks.append({**section, "text": section_text})
            continue
        start = 0
        part = 0
        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text.strip(),
                    "page": section.get("page"),
                    "source": section["source"],
                    "part": part,
                })
                part += 1
            start += chunk_size - overlap
    return chunks
