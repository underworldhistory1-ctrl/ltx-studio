# 🎬 LTX 2.5 Video Enhancement Studio (ReDetail 2.0)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/pytorch-2.5%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CUDA 12.1+ / 13.0](https://img.shields.io/badge/CUDA-12.1%20%7C%2013.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)

An enterprise-grade, turnkey **Video-to-Video (V2V) Enhancement, Upscaling, and Restoration Studio** powered by **Lightricks LTX-2.5 22B Distilled DiT**, **Gemma 4 12B**, **ReDetail 2.0**, and **In-Context LoRAs (IC-LoRA)**. Built for modern high-performance GPUs (NVIDIA RTX 5090 Blackwell, RTX 4090, A100, H100).

---

## 🌟 Key Capabilities

* **ReDetail 2.0 Cinematic Pipeline**:
  * **Refine Details Mode**: Sharpens fine textures, enhances facial clarity, micro-expressions, hair, and clothing while preserving 100% actor identity and framing.
  * **Restore & Colorize Mode**: Cleans heavy compression noise, banding, scratches, and archival artifacts before detail enhancement.
  * **Pixel Spatial Upscaler x2**: Rapid high-frequency super-resolution mode for clean footage.
* **4K & Multi-Resolution Tiled Fusion**:
  * Tiled latent spatial and temporal sampling allowing 1080p, 2K, and 4K enhancement without out-of-memory errors.
  * Adaptive overlap blending (`blend_var`) preventing edge seams and boundary flicker.
* **Smart Multi-Stack Queue Coordination**:
  * Built-in mutual exclusion detecting active renders on shared GPU environments (such as **MiniMax H3** on port `8188`).
  * Automatically pauses LTX renders in queue, monitors GPU memory, flushes VRAM cache, and seamlessly resumes when the GPU becomes available.
* **Lossless Audio Preservation**:
  * Extracts multi-channel audio from source footage, calculates exact temporal alignment to $8n+1$ latent frame boundaries, and remuxes losslessly via FFmpeg.
* **Modern Web Interface**:
  * Interactive **Before/After Split-Screen Video Slider** to inspect enhancements frame-by-frame.
  * Real-time hardware telemetry: GPU utilization, VRAM usage, temperature, CPU/RAM usage.
  * Live generation progress bar, stage descriptions, and live frame preview streaming over SSE / WebSockets.
  * Persistent Job History with one-click re-queue, download, cancel, and inspect controls.
* **One-Click Turnkey Launcher**:
  * Automatic dependency resolution, repository cloning, virtual environment management, model weight downloading (via high-speed ModelScope with Hugging Face fallback), and process supervision.

---

## 🚀 Quickstart (One Command)

### 1. Clone & Launch
```bash
git clone https://github.com/underworldhistory1-ctrl/ltx-studio.git
cd ltx-studio

# Run the turnkey launcher (auto-installs requirements, models, and starts studio)
python3 launcher.py
```

Or using the bash script:
```bash
bash install_and_run.sh
```

On Windows:
```cmd
run.bat
```

### 2. Access the Studio
* **Local Browser**: [http://localhost:7860](http://localhost:7860)
* **Internal ComfyUI Engine**: `http://127.0.0.1:8189`
* **Remote Cloud GPU**:
  ```bash
  ssh -L 7860:127.0.0.1:7860 root@<your-cloud-ip> -p <port>
  ```
  Then open `http://localhost:7860` in your local browser.

---

## ⚙️ Launcher Options & Subcommands

The unified `launcher.py` handles installation, model acquisition, and execution:

| Command / Flag | Description |
| :--- | :--- |
| `python launcher.py` | Full auto-pilot: validates environment, downloads missing models, and launches Studio + ComfyUI worker. |
| `python launcher.py --daemon` | Starts Studio server in the background and returns to shell. |
| `python launcher.py --install-only` | Runs pre-flight checks, creates virtualenv, clones nodes, downloads all models, and exits without launching. |
| `python launcher.py --run-only` | Skips installer verification and launches services immediately (fast boot). |
| `python launcher.py --status` | Prints live status of Web Server, ComfyUI Worker, GPU memory, and Queue. |
| `python launcher.py --stop` | Gracefully terminates all background Studio and ComfyUI worker processes. |
| `python launcher.py --restart` | Cleanly restarts all services. |
| `python launcher.py --port 7860` | Specify custom Web UI port (default: `7860`). |
| `python launcher.py --comfy-port 8189` | Specify custom internal ComfyUI worker port (default: `8189`). |
| `python launcher.py --reserve-vram 2.0` | Reserve VRAM headroom in GB to prevent system OOM (default: `2.0`). |
| `python launcher.py --with-gguf` | Download optional 15.7 GB GGUF quantized model for low-VRAM GPUs. |
| `python launcher.py --hf-token "hf_..."` | Provide Hugging Face access token for gated repositories (ModelScope mirror is used by default). |

---

## 🏗️ Repository Architecture

```
ltx-studio/
├── launcher.py               # Unified CLI & supervisor entrypoint
├── install_and_run.sh        # Bash wrapper script
├── run.bat                   # Windows batch launcher
├── requirements.txt          # Python dependencies
├── studio/                   # LTX Studio Web Application & Engine
│   ├── server.py             # Aiohttp REST server & SSE stream (Port 7860)
│   ├── engine.py             # ReDetail 2.0 execution orchestrator & FFmpeg remuxer
│   ├── jobs.py               # Job queue manager & MiniMax H3 mutual exclusion
│   ├── graph.py              # Dynamic ComfyUI API graph compiler
│   ├── settings.py           # Single source of truth settings schema & presets
│   ├── comfy.py              # ComfyUI worker supervisor & WebSocket/HTTP client
│   ├── video.py              # Video probing, scene cut detection & thumbnail extractor
│   ├── colorfix.py           # Color consistency & temporal seam matcher
│   ├── sysmon.py             # Real-time GPU & system telemetry
│   ├── common.py             # Shared paths and configuration
│   └── static/               # Frontend Assets
│       ├── index.html        # Modern Dark-Mode UI & Split-Screen Video Player
│       ├── app.js            # Reactive UI controller & SSE listeners
│       └── style.css         # Glassmorphism styling & responsive layout
├── installer/
│   ├── install.py            # Automated multi-phase installer with resume support
│   └── manifest.json         # Master model hashes, URLs & repository pinned commits
├── vendor/
│   └── redetail/             # ReDetail 2.0 core engine & conditioning cache nodes
├── config/
│   ├── studio.json           # Studio server settings
│   └── extra_model_paths.yaml# ComfyUI model mapping
├── models/                   # Model weight storage (git-ignored)
│   ├── diffusion_models/     # LTX-2.5 22B Distilled Transformer INT8
│   ├── text_encoders/        # Gemma-4 12B INT8 ConvRot
│   ├── vae/                  # Video & Audio VAEs
│   ├── loras/                # Refine, Restore, and Upscaler IC-LoRAs
│   └── unet/                 # Optional GGUF weights
└── logs/                     # Application logs (studio.log, comfy.log)
```

---

## 📦 Required Models & Weights

The installer automatically fetches and validates these models via ModelScope / Hugging Face:

| Model ID | File Name | Target Directory | Size | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `dit_int8` | `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | `models/diffusion_models/` | 21.5 GB | Core Distilled DiT |
| `te_int8` | `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | `models/text_encoders/` | 15.4 GB | Text Encoder & Projection |
| `vae_video` | `ltx-2.5-video-vae-bf16.safetensors` | `models/vae/` | 1.47 GB | Video VAE Decoder/Encoder |
| `vae_audio` | `ltx-2.5-audio-vae-bf16.safetensors` | `models/vae/` | 365 MB | Audio Latent VAE |
| `lora_refine`| `ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors` | `models/loras/` | 1.31 GB | ReDetail High-Frequency LoRA |
| `lora_pixel` | `ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors` | `models/loras/` | 327 MB | Spatial Super-Resolution LoRA |
| `lora_restore`| `ltx-2.5-22b-ic-lora-restore-1.0.safetensors` | `models/loras/` | 1.71 GB | Denoise & Artifact Cleanup |

---

## 🤝 Smart Queue Coordination with MiniMax H3

When deployed on a shared GPU instance alongside a MiniMax H3 server running on port `8188`:
1. **Zero Memory Collisions**: The Studio automatically pings `http://127.0.0.1:8188/queue`. If H3 is actively generating a clip, LTX holds the incoming enhancement job in `queued` status (`⏳ IN QUEUE (#1)`).
2. **Automatic Execution**: The moment the H3 generation completes, LTX flushes PyTorch CUDA cache (`POST /free`), claims the GPU, and begins rendering without user intervention.
3. **Resilience**: If a job is interrupted by a server restart or user action, a one-click `🔄 Re-queue` button allows immediate resumption.

---

## ⚡ NVIDIA RTX 5090 (Blackwell) & RTX 4090 Optimizations

On NVIDIA Blackwell and Ada Lovelace architectures, large DiT transformer allocations can experience memory fragmentation. This studio applies proven production settings:
```python
# Set in worker supervisor environment
export PYTORCH_CUDA_ALLOC_CONF="backend:native,expandable_segments:True"
```
Cache policy is configurable. `cache_mode: "none"` keeps conservative per-chunk cleanup. For repeated renders, adaptive RAM caching can retain loaders and conditioning while evicting tensors under memory pressure; the engine preserves that cache instead of resetting the executor before and after every chunk. On a 40 GB RAM / RTX 5090 deployment, `cache_mode: "auto"` with `comfy_extra_args: ["--cache-ram", "8", "16"]` was selected for bounded caching. Recheck available RAM and worker flag support before using those thresholds elsewhere. This does not guarantee that every 4K workload fits memory.

A short repeatability check on that deployment (512x288,49 frames,24fps,Refine,8 steps,seed42) completed in67.2 seconds cold and15.1 seconds warm. ComfyUI history confirmed reuse of model/VAE/IC-LoRA/conditioning loaders while sampling reran. The output retained49 frames and audio and passed full FFmpeg decoding; no cgroup OOM kills occurred. These timings are specific to the small repeated workload, not full-resolution throughput guarantees.

GPU admission counts only the matching LTX worker's resident VRAM as reusable. An idle H3 queue is verified before requesting its cache release, avoiding a cached-model handoff deadlock. Idle release timers are serialized against jobs and recheck the actual ComfyUI queue, so an old timer cannot clear the next job's cache. The extra model paths file resolves `../models` relative to its config directory, keeping cloned installations portable.

---

## 📡 REST API Reference

The browser refreshes job snapshots every five seconds as a fallback to live SSE updates, and refreshes immediately after reconnection or returning to the tab. A `?job=ltx_...` link and per-tab selection restore the tracked job after reload; a fresh page can adopt an ongoing server render. Uploading a new source leaves that draft in the editor without cancelling existing work. The queue badge includes the active LTX render.

Submission and re-queue requests validate HTTP responses and job receipts, report gateway HTML without exposing a JSON parser error, and time out after 45 seconds while retaining the current source and settings. Quick Test and Full Pass cannot submit concurrently. An unconfirmed POST is never automatically repeated: check Queue & History before resubmitting because the server may already have accepted it.

On `*.trycloudflare.com`, the frontend uses short JSON polling for both job and installer progress instead of opening SSE connections, which [Quick Tunnels do not support](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/). Installer snapshot requests have a 10-second deadline and coalesce while pending. Other hosts retain SSE with job polling as a fallback.

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/upload` | `POST` | Upload a video file (`multipart/form-data`) |
| `/api/probe` | `POST` | Probe video metadata, dimensions, audio, and scene cuts |
| `/api/jobs` | `POST` | Submit a new enhancement job with custom settings |
| `/api/jobs` | `GET` | List all historical, queued, and completed jobs |
| `/api/jobs/{id}` | `GET` | Get detailed state, chunk progress, and output URL |
| `/api/jobs/{id}/cancel` | `POST` | Cancel a running or queued job |
| `/api/jobs/{id}/requeue`| `POST` | Re-queue an interrupted or completed job |
| `/api/jobs/{id}` | `DELETE`| Remove job record and associated temporary files |
| `/api/status` | `GET` | System health, GPU telemetry, VRAM metrics, and queue status |
| `/api/presets` | `GET` | Retrieve available presets (`cinematic_4k`, `portrait_clean`, etc.) |
| `/api/models` | `GET` | List installed and missing model weights |

---

## 📄 License
This project is released under the **MIT License**. Underlying model weights are subject to the Lightricks LTX-2.5 and Google Gemma research licenses.
