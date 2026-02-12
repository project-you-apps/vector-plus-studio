"""
Cartridge I/O -- load, save, list, validate.

Extracted from vector_plus_studio_v83.py lines 265-397.
"""

import os
import pickle
import hashlib
import json
import numpy as np

from .engine import engine, TextRegionEncoder


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cartridges")
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")


def get_cartridge_dirs() -> list[str]:
    """Return list of directories to scan for cartridges."""
    dirs = [DATA_DIR]
    if os.path.isdir(SAMPLE_DIR):
        dirs.append(SAMPLE_DIR)
    return dirs


def list_cartridges() -> list[dict]:
    """List all available cartridges across cartridge dirs."""
    results = []
    seen = set()

    for d in get_cartridge_dirs():
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            if not f.endswith(".pkl"):
                continue
            if f in seen:
                continue
            seen.add(f)

            name = os.path.splitext(f)[0]
            path = os.path.join(d, f)
            size_mb = os.path.getsize(path) / (1024 * 1024)

            brain_path = os.path.join(d, f"{name}_brain.npy")
            sig_path = os.path.join(d, f"{name}_signatures.npz")
            manifest_path = os.path.join(d, f"{name}_brain_manifest.json")

            # Also check DATA_DIR for companion files if pkl is in SAMPLE_DIR
            if d == SAMPLE_DIR:
                if not os.path.exists(brain_path) and os.path.exists(os.path.join(DATA_DIR, f"{name}_brain.npy")):
                    brain_path = os.path.join(DATA_DIR, f"{name}_brain.npy")
                if not os.path.exists(sig_path) and os.path.exists(os.path.join(DATA_DIR, f"{name}_signatures.npz")):
                    sig_path = os.path.join(DATA_DIR, f"{name}_signatures.npz")
                if not os.path.exists(manifest_path) and os.path.exists(os.path.join(DATA_DIR, f"{name}_brain_manifest.json")):
                    manifest_path = os.path.join(DATA_DIR, f"{name}_brain_manifest.json")

            results.append({
                "name": name,
                "filename": f,
                "path": path,
                "size_mb": round(size_mb, 1),
                "has_brain": os.path.exists(brain_path),
                "has_signatures": os.path.exists(sig_path),
                "has_manifest": os.path.exists(manifest_path),
                "brain_path": brain_path,
                "sig_path": sig_path,
            })

    # Also find brain-only (no pkl)
    for d in get_cartridge_dirs():
        for f in os.listdir(d):
            if f.endswith("_brain.npy"):
                name = f.replace("_brain.npy", "")
                if name + ".pkl" not in seen and f"{name} (brain only)" not in seen:
                    seen.add(f"{name} (brain only)")
                    brain_path = os.path.join(d, f)
                    sig_path = os.path.join(d, f"{name}_signatures.npz")
                    results.append({
                        "name": name,
                        "filename": f"{name} (brain only)",
                        "path": None,
                        "size_mb": round(os.path.getsize(brain_path) / (1024 * 1024), 1),
                        "has_brain": True,
                        "has_signatures": os.path.exists(sig_path),
                        "has_manifest": False,
                        "brain_path": brain_path,
                        "sig_path": sig_path,
                    })

    return results


def load_cartridge(path: str) -> dict | None:
    """Load cartridge -- supports legacy and multimodal formats."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        version = data.get("version", "0")

        if version == "8.3":
            return {
                'embeddings': np.array(data['embeddings']),
                'passages': data['passages'],
                'compressed_lens': data.get('compressed_lens', []),
                'multimodal': True,
            }

        if version in ["7.0", "8.0", "8.1", "8.2"]:
            core = data.get("data", data)
            emb = core.get("embeddings", data.get("embeddings"))
            txt = core.get("passages", data.get("passages", []))
            return {
                'embeddings': np.array(emb) if emb is not None else None,
                'passages': list(txt),
                'multimodal': False,
            }

        if "data" in data:
            core = data["data"]
            return {
                'embeddings': np.array(core.get("embeddings")),
                'passages': core.get("passages", []),
                'multimodal': False,
            }

        if "embeddings" in data:
            return {
                'embeddings': np.array(data["embeddings"]),
                'passages': data.get("passages", data.get("texts", [])),
                'multimodal': False,
            }

    return None


def save_cartridge_multimodal(path: str, embeddings, passages, compressed_lens):
    """Save multimodal cartridge with compressed lengths."""
    cart = {
        "version": "8.3",
        "embeddings": embeddings,
        "passages": passages,
        "compressed_lens": compressed_lens,
    }
    with open(path, "wb") as f:
        pickle.dump(cart, f)
    return True


def load_signatures(sig_path: str) -> dict | None:
    if not os.path.exists(sig_path):
        return None
    try:
        data = np.load(sig_path, allow_pickle=True)
        result = {
            'signatures': data['signatures'],
            'titles': data['titles'] if 'titles' in data else None,
            'n_patterns': int(data['n_patterns']) if 'n_patterns' in data else len(data['signatures']),
            'compressed_lens': data['compressed_lens'] if 'compressed_lens' in data else None,
            'signature_method': str(data['signature_method']) if 'signature_method' in data else 'legacy',
        }
        if 'compressed_texts' in data:
            result['compressed_texts'] = list(data['compressed_texts'])
        return result
    except Exception as e:
        print(f"[Signatures] Failed to load {sig_path}: {e}")
        return None


def save_signatures(sig_path, signatures, titles=None, compressed_lens=None,
                    compressed_texts=None, signature_method="l3"):
    save_dict = {
        'pattern_ids': np.arange(len(signatures), dtype=np.int32),
        'signatures': signatures,
        'n_patterns': len(signatures),
        'signature_dim': signatures.shape[1] if len(signatures.shape) > 1 else 65536,
        'signature_method': np.array(signature_method),
    }
    if titles is not None:
        save_dict['titles'] = np.array(titles, dtype=object)
    if compressed_lens is not None:
        save_dict['compressed_lens'] = np.array(compressed_lens, dtype=np.int32)
    if compressed_texts is not None:
        save_dict['compressed_texts'] = np.array(compressed_texts, dtype=object)
    np.savez_compressed(sig_path, **save_dict)
    return sig_path


def compute_cartridge_fingerprint(embeddings):
    count = len(embeddings)
    first_bytes = embeddings[0].tobytes()
    last_bytes = embeddings[-1].tobytes() if count > 1 else first_bytes
    combined = first_bytes + last_bytes + str(count).encode()
    fingerprint = hashlib.sha256(combined).hexdigest()[:16]
    return {"count": count, "fingerprint": fingerprint}


def save_brain_manifest(brain_path, embeddings):
    manifest_path = brain_path.replace("_brain.npy", "_brain_manifest.json")
    manifest = compute_cartridge_fingerprint(embeddings)
    manifest["version"] = "8.3"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    return manifest_path


def validate_brain_manifest(brain_path, embeddings):
    manifest_path = brain_path.replace("_brain.npy", "_brain_manifest.json")
    if not os.path.exists(manifest_path):
        return True, "Legacy brain (no manifest)"
    try:
        with open(manifest_path, "r") as f:
            saved_manifest = json.load(f)
        current = compute_cartridge_fingerprint(embeddings)
        if saved_manifest["count"] != current["count"]:
            return False, f"Count mismatch ({saved_manifest['count']} vs {current['count']})"
        if saved_manifest["fingerprint"] != current["fingerprint"]:
            return False, "Fingerprint mismatch"
        return True, "Manifest validated"
    except Exception as e:
        return False, f"Manifest error: {e}"


def find_cartridge_path(filename: str) -> str | None:
    """Find full path for a cartridge filename across all dirs."""
    for d in get_cartridge_dirs():
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    return None


def find_companion_file(name: str, suffix: str) -> str | None:
    """Find companion file (brain, signatures) across dirs."""
    for d in get_cartridge_dirs():
        path = os.path.join(d, f"{name}{suffix}")
        if os.path.exists(path):
            return path
    # Also check DATA_DIR explicitly
    path = os.path.join(DATA_DIR, f"{name}{suffix}")
    if os.path.exists(path):
        return path
    return None
