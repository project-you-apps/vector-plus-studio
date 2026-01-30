# Vector+ Studio v0.81

**Physics-Enhanced Semantic Search**

Vector+ Studio is a semantic search application powered by a 16-million neuron Hopfield network. Unlike traditional vector databases, queries go through **real neural physics** - the lattice actively shapes search results through associative memory, not just cosine similarity.

![Vector+ Studio Screenshot](docs/screenshot.png)

## What's New in v0.81

**Vector+ Studio v0.81** introduces **physics-enhanced search** - the neural lattice now actively participates in ranking, not just visualization.

### Key Breakthrough: Holographic Storage

The trained "brain" file is **fixed at 128MB regardless of dataset size**:

| Dataset | Embeddings File | Brain File |
|---------|-----------------|------------|
| 10k articles | ~30 MB | **128 MB** |
| 100k articles | ~300 MB | **128 MB** |
| 500k articles | ~1.5 GB | **128 MB** |

This is how biological memory works - patterns are stored holographically in synaptic weights, not as separate records. The same 128MB weight matrix can encode 10k or 500k patterns.

## Features

- **Physics-Enhanced Search**: Queries go through encode → settle → decode pipeline; Hebbian weights shape results
- **Holographic Brain Storage**: 128MB brain file stores unlimited patterns (fixed size!)
- **Background Training**: Train on large datasets without blocking the UI
- **Brain Persistence**: Save/load trained weights - instant remount after first training
- **Neural Lattice Visualization**: See query and result patterns on a 4096x4096 neuron lattice
- **Noise Tolerance**: 86%+ correlation even with 30% input corruption
- **Full-Text Keyword Boost**: Phrase matching in document bodies, not just titles
- **Document Ingestion**: Import PDFs, Word docs, and text files
- **Memory Cartridges**: Save and load document collections

## Quick Start

### Requirements

- Windows 10/11
- NVIDIA GPU with CUDA support
- Python 3.10+

### Installation

```bash
git clone https://github.com/project-you-apps/vector-plus-studio.git
cd vector-plus-studio
pip install -r requirements.txt
```

### Run

```bash
streamlit run vector_plus_studio_v81.py --server.fileWatcherType none
```

Then open http://localhost:8501 in your browser.

**Note:** The `--server.fileWatcherType none` flag prevents unnecessary reloads during use.

### First Search

The first search will take ~30 seconds while the embedding model (Nomic Embed v1.5) downloads and loads. Subsequent searches are fast (~500ms).

## How It Works

1. **Thermometer Encoding**: Each embedding dimension maps to a 64x64 region of the lattice
2. **Hebbian Training**: Patterns are learned into synaptic weights during cartridge mount
3. **Physics-Enhanced Query**: Query embedding is encoded → settled (physics shapes it) → decoded back
4. **Associative Search**: The "physics-cleaned" embedding finds semantic neighborhoods, not just exact matches
5. **Keyword Boosting**: Full-text phrase matching re-ranks top candidates

## Project Structure

```
vector-plus-studio/
├── vector_plus_studio_v81.py     # Main Streamlit application
├── multi_lattice_wrapper_v7.py  # Python wrapper for CUDA engine
├── thermometer_encoder_generic_64x64.py  # Encoding utilities
├── bin/
│   └── lattice_cuda_v7.dll      # Pre-built CUDA physics engine
├── cartridges/                   # Your saved document collections
└── sample_data/                  # Sample datasets
```

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GTX 1060 | NVIDIA RTX 3080+ |
| VRAM | 4 GB | 8+ GB |
| RAM | 8 GB | 16+ GB |
| CUDA | 11.0+ | 12.0+ |

## License

**Dual-Licensed:**

| Component | License | Commercial Use |
|-----------|---------|----------------|
| Python code (`.py` files) | MIT | Yes |
| CUDA Engine (`bin/*.dll`) | Proprietary | [Contact for license](mailto:licensing@project-you.app) |

The Python wrapper and utilities are open source under MIT. The compiled CUDA physics engine is free for personal, educational, and non-commercial use. Commercial use requires a separate license - see [bin/LICENSE](bin/LICENSE).

## Links

- [Project You](https://project-you.app) - Parent project

## Future Direction and Updates

- **v0.81 (Current)**: Physics-enhanced search is now live! Queries go through real neural physics.
- **Coming Soon**: GPU-accelerated embedding for <100ms total query time
- **Planned**: Multi-brain support (load multiple trained domains), larger lattice sizes for higher capacity

---

Built with physics, not just math. Patterns stored holographically, not as records.
