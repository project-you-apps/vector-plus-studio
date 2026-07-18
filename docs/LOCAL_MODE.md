# Vector+ Studio — Local Mode

> Last updated: 2026-05-30. This doc is **scope A** of the local-mode pass — the documentation half. The Docker-packaging half (scope B) is queued separately.

## What "local mode" means

When you run VPS on your own machine, the "server" filesystem **is** your filesystem. The same `FolderPickerModal` that walks the droplet's disk in the hosted deployment walks your own disk in local mode. There is no client-server upload step for local files; the folder picker, cart browser, and ingestion source pickers all browse your machine directly via the existing `/api/cartbuilder/browse` endpoint.

This means:

- **Adding a saved-folder root for the cart browser** → picks a folder on your machine, not on the droplet
- **Choosing an ingestion source directory** → reads files from your disk for cart building
- **Mounting an existing cart** → loads a `.cart.npz` from anywhere on your machine
- **Opening "Edit Carts"** → operates against your local cart library

The hosted droplet keeps its upload-via-browser UX for users who don't want to install anything. Local mode is for users who want their files to stay on their own machine.

## Why local mode matters

Three reasons users would want local-mode VPS over the hosted droplet:

1. **Files never leave your machine.** Internal documents, confidential research, personal notes — everything stays on disk.
2. **No upload size cap.** The droplet has practical limits on file count and total bytes; your local disk does not.
3. **Your own GPU.** If you have CUDA available, the local FastAPI server uses your hardware for physics rather than competing for the droplet's GPU.

## How to run VPS locally

The current local entry point is the FastAPI dev server in `vector-plus-studio-repo/api/`:

```powershell
# from the vector-plus-studio-repo directory
uvicorn api.main:app --reload --port 8000
```

Then start the React frontend separately (default Vite dev server on `:5173` proxies API calls to `:8000`):

```powershell
# from vector-plus-studio-repo/frontend
npm run dev
```

Open `http://localhost:5173` in your browser. The "server" the UI talks to is your own `uvicorn` process; the folder picker walks your local disk.

## How the folder picker works

The component lives at [`frontend/src/components/FolderPickerModal.tsx`](../frontend/src/components/FolderPickerModal.tsx). Its header comment notes:

> server-side path browser. Used to add saved-folder roots for the cart browser, and to pick ingestion source dirs. Walks the actual server filesystem via `/api/cartbuilder/browse`.

In hosted mode, "server filesystem" = droplet disk. In local mode, "server filesystem" = your machine. Same code path; the deployment context determines what disk is walked.

## Caveats and known gaps

- **No Docker image yet.** Users currently need Python + Node installed locally. A Dockerfile is on the backlog (scope B of the local-mode pass).
- **Sample carts.** The bundled sample cartridges (LatticeRunner, Project You overview, etc.) ship with the droplet. To work with them locally, copy the `.cart.npz` files from your droplet checkout into a local directory and add it as a saved-folder root.
- **OAuth flows are droplet-only.** Local mode runs without authentication; the auth router assumes the public cookie domain. If you need per-user libraries locally, use single-user mode (no login UI) and keep carts in directories you own.
- **Persistent memory.** See [`PERSISTENT-MEMORY-SETUP.md`](PERSISTENT-MEMORY-SETUP.md) for how Mempack and saved searches persist in local mode versus the droplet.

## What's NOT in this doc

- **Packaging as an .exe or installer.** Downstream scope — open question whether to ship a single-file standalone or a personal-infra setup.
- **WebGPU vs CUDA tradeoffs.** WebGPU is the browser-build path (cart builder); CUDA is the physics path. Local mode uses CUDA if available, browser WebGPU otherwise.
