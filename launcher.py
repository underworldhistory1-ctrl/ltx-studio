#!/usr/bin/env python3
"""
🎬 LTX 2.5 Video Enhancement Studio — Turnkey Launcher
=====================================================
One-command installation, model acquisition, and unified service supervision
for the LTX 2.5 Video Enhancement & Restoration Studio (ReDetail 2.0 Engine).

Usage:
    python launcher.py                  # Auto-install (if needed) & run in foreground
    python launcher.py --daemon         # Start in background
    python launcher.py --install-only   # Install dependencies & download models, then exit
    python launcher.py --run-only       # Start services directly
    python launcher.py --status         # Display live status of services & GPU
    python launcher.py --stop           # Gracefully stop all Studio services
    python launcher.py --restart        # Restart all services
"""

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# Fix Windows console UTF-8 encoding for unicode characters
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Paths
ROOT = Path(__file__).resolve().parent
STUDIO_DIR = ROOT / "studio"
INSTALLER_DIR = ROOT / "installer"
CONFIG_DIR = ROOT / "config"
LOGS_DIR = ROOT / "logs"
STATE_DIR = ROOT / "state"
MODELS_DIR = ROOT / "models"

for d in (CONFIG_DIR, LOGS_DIR, STATE_DIR, MODELS_DIR):
    if d == MODELS_DIR and d.is_symlink() and not d.exists():
        d.resolve().mkdir(parents=True, exist_ok=True)
    d.mkdir(parents=True, exist_ok=True)

# Ensure environment knows its root
os.environ["LTX_STUDIO_ROOT"] = str(ROOT)

BANNER = r"""
================================================================================
   🎬  LTX 2.5 VIDEO ENHANCEMENT STUDIO (ReDetail 2.0 Engine)
   Turnkey Video-to-Video Upscaling & Enhancement with Smart GPU Queue
================================================================================
"""


def log(msg, level="INFO"):
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    prefix = {
        "INFO": "\033[1;34m[INFO]\033[0m",
        "SUCCESS": "\033[1;32m[SUCCESS]\033[0m",
        "WARN": "\033[1;33m[WARN]\033[0m",
        "ERROR": "\033[1;31m[ERROR]\033[0m",
    }.get(level, f"[{level}]")
    print(f"{t} {prefix} {msg}", flush=True)


def get_python_exe():
    """Finds the best Python interpreter to use."""
    # 1. Repository private venv
    venv_py = ROOT / "venv" / ("bin" if platform.system() != "Windows" else "Scripts") / ("python" if platform.system() != "Windows" else "python.exe")
    if venv_py.exists():
        return str(venv_py)
    # 2. Host virtualenv
    for cand in ("/opt/venv/bin/python", "/workspace/venv/bin/python"):
        if os.path.exists(cand):
            return cand
    # 3. Conda environment
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        cand = Path(conda_prefix) / ("bin" if platform.system() != "Windows" else "Scripts") / ("python" if platform.system() != "Windows" else "python.exe")
        if cand.exists():
            return str(cand)
    # 4. Current interpreter
    return sys.executable


def get_disk_free_gb(path):
    try:
        usage = shutil.disk_usage(str(path))
        return usage.free / (1024 ** 3)
    except Exception:
        return 0.0


def check_gpu():
    """Detects NVIDIA GPU and VRAM."""
    gpu_info = {"present": False, "name": "Unknown", "vram_mb": 0}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0 and r.stdout.strip():
            first = r.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            gpu_info["present"] = True
            gpu_info["name"] = parts[0]
            if len(parts) > 1:
                gpu_info["vram_mb"] = float(parts[1])
    except Exception:
        pass
    return gpu_info


def check_is_installed():
    """Checks whether the core repo and models are installed."""
    comfy_dir = ROOT / "ComfyUI"
    dit_model = MODELS_DIR / "diffusion_models" / "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
    return comfy_dir.exists() and dit_model.exists() and dit_model.stat().st_size > 10_000_000


def update_extra_model_paths():
    """Writes extra_model_paths.yaml pointing to the current repository root."""
    target = CONFIG_DIR / "extra_model_paths.yaml"
    content = (
        "ltx_studio:\n"
        f"    base_path: {ROOT.as_posix()}/models\n"
        "    diffusion_models: diffusion_models\n"
        "    text_encoders: text_encoders\n"
        "    clip: text_encoders\n"
        "    vae: vae\n"
        "    loras: loras\n"
        "    unet: unet\n"
    )
    target.write_text(content)


def load_studio_config():
    p = CONFIG_DIR / "studio.json"
    default = {
        "host": "0.0.0.0",
        "port": 7860,
        "comfy_port": 8189,
        "reserve_vram_gb": 2.0,
        "attention": "ck-int8",
        "cache_mode": "ram",
        "coordinate_h3": True,
        "h3_url": "http://127.0.0.1:8188",
        "release_vram_idle_s": 90,
        "auto_resume_jobs": True,
        "comfy_extra_args": []
    }
    if p.exists():
        try:
            cfg = json.loads(p.read_text())
            default.update(cfg)
        except Exception:
            pass
    return default


def save_studio_config(cfg):
    p = CONFIG_DIR / "studio.json"
    p.write_text(json.dumps(cfg, indent=2))


def check_h3_status(h3_url):
    """Checks if MiniMax H3 is reachable on its configured URL."""
    try:
        import urllib.request
        req = urllib.request.Request(f"{h3_url}/queue", headers={"User-Agent": "LTX-Studio-Launcher"})
        with urllib.request.urlopen(req, timeout=1.5) as r:
            if r.status == 200:
                data = json.loads(r.read().decode())
                executing = len(data.get("queue_running", [])) > 0
                return True, executing
    except Exception:
        pass
    return False, False


def is_pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def find_pid_by_port(port):
    """Finds process ID listening on a given port."""
    import re
    try:
        r = subprocess.run(["ss", "-tulpn"], capture_output=True, text=True, timeout=2)
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                if f":{port} " in line or f":{port}\t" in line:
                    match = re.search(r"pid=(\d+)", line)
                    if match:
                        return int(match.group(1))
    except Exception:
        pass
    try:
        r = subprocess.run(["lsof", "-t", "-i", f":{port}"], capture_output=True, text=True, timeout=2)
        if r.returncode == 0 and r.stdout.strip():
            return int(r.stdout.strip().split()[0])
    except Exception:
        pass
    return None


def get_running_pids():
    studio_pid = None
    comfy_pid = None

    p1 = STATE_DIR / "studio.pid"
    if p1.exists():
        try:
            val = int(p1.read_text().strip())
            if is_pid_alive(val):
                studio_pid = val
        except Exception:
            pass

    p2 = STATE_DIR / "comfy.pid"
    if p2.exists():
        try:
            d = json.loads(p2.read_text())
            val = d.get("pid")
            if val and is_pid_alive(val):
                comfy_pid = val
        except Exception:
            pass

    cfg = load_studio_config()
    if not studio_pid:
        p = find_pid_by_port(cfg.get("port", 7860))
        if p and is_pid_alive(p):
            studio_pid = p
            (STATE_DIR / "studio.pid").write_text(str(p))

    if not comfy_pid:
        p = find_pid_by_port(cfg.get("comfy_port", 8189))
        if p and is_pid_alive(p):
            comfy_pid = p
            (STATE_DIR / "comfy.pid").write_text(json.dumps({"pid": p}))

    return studio_pid, comfy_pid


def cmd_stop():
    """Stops Studio server and ComfyUI worker."""
    studio_pid, comfy_pid = get_running_pids()
    log("Stopping LTX Video Enhancement Studio services...")

    if studio_pid:
        log(f"Terminating Studio Server (PID {studio_pid})...")
        try:
            os.kill(studio_pid, signal.SIGTERM)
        except Exception:
            pass
        for _ in range(10):
            if not is_pid_alive(studio_pid):
                break
            time.sleep(0.3)
        if is_pid_alive(studio_pid):
            try:
                os.kill(studio_pid, signal.SIGKILL)
            except Exception:
                pass
        (STATE_DIR / "studio.pid").unlink(missing_ok=True)
        log("Studio Server stopped.", "SUCCESS")
    else:
        log("Studio Server is not running.")

    if comfy_pid:
        log(f"Terminating ComfyUI Worker (PID {comfy_pid})...")
        try:
            os.kill(comfy_pid, signal.SIGTERM)
        except Exception:
            pass
        for _ in range(10):
            if not is_pid_alive(comfy_pid):
                break
            time.sleep(0.3)
        if is_pid_alive(comfy_pid):
            try:
                os.kill(comfy_pid, signal.SIGKILL)
            except Exception:
                pass
        (STATE_DIR / "comfy.pid").unlink(missing_ok=True)
        log("ComfyUI Worker stopped.", "SUCCESS")
    else:
        log("ComfyUI Worker is not running.")

    # Terminate any lingering install script
    try:
        subprocess.run(["pkill", "-f", f"install.py --root {ROOT}"], stderr=subprocess.DEVNULL)
    except Exception:
        pass


def cmd_status():
    """Prints live status report."""
    print(BANNER)
    studio_pid, comfy_pid = get_running_pids()
    cfg = load_studio_config()

    print(f"📁 Repository Root:   {ROOT}")
    print(f"🐍 Python Executable: {get_python_exe()}")
    print(f"💾 Free Disk Space:  {get_disk_free_gb(ROOT):.1f} GB")

    gpu = check_gpu()
    if gpu["present"]:
        print(f"🎮 GPU:               {gpu['name']} ({gpu['vram_mb']:.0f} MB total)")
    else:
        print("🎮 GPU:               No NVIDIA GPU detected")

    print("-" * 80)
    print(f"🌐 Studio Web Server: {'RUNNING (PID ' + str(studio_pid) + ')' if studio_pid else 'STOPPED'}")
    print(f"   Listening on:      http://{cfg['host']}:{cfg['port']}")

    print(f"⚙️ ComfyUI Worker:    {'RUNNING (PID ' + str(comfy_pid) + ')' if comfy_pid else 'STOPPED'}")
    print(f"   Internal Port:     http://127.0.0.1:{cfg['comfy_port']}")

    # Shared stack coordination
    h3_reachable, h3_busy = check_h3_status(cfg["h3_url"])
    if h3_reachable:
        state_str = "BUSY (Rendering)" if h3_busy else "IDLE (Ready)"
        print(f"🤝 MiniMax H3 Stack:  DETECTED at {cfg['h3_url']} [{state_str}]")
        print("   Smart GPU Queue:   ACTIVE (Mutual exclusion enabled)")
    else:
        print(f"🤝 MiniMax H3 Stack:  Not detected at {cfg['h3_url']} (Standalone mode)")

    # Install status
    inst_file = STATE_DIR / "install.json"
    if inst_file.exists():
        try:
            d = json.loads(inst_file.read_text())
            st = d.get("status", "unknown").upper()
            pct = d.get("overall", {}).get("pct", 0)
            print(f"📦 Installer State:   {st} ({pct}%)")
        except Exception:
            pass
    print("=" * 80)


def run_installer(args):
    """Executes installer/install.py synchronously with terminal output."""
    log("Running automated installation and model checks...")
    py = get_python_exe()
    installer_script = INSTALLER_DIR / "install.py"

    cmd = [py, str(installer_script), "--root", str(ROOT)]
    if args.no_gguf:
        cmd.append("--no-gguf")
    else:
        cmd.append("--with-gguf")
    if args.skip_models:
        cmd.append("--skip-models")
    if args.verify_only:
        cmd.append("--verify-only")

    if args.hf_token:
        (STATE_DIR / "hf.token").write_text(args.hf_token.strip())

    log(f"Command: {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        log("Installer exited with errors. Check logs/installer.log or state/install.json for details.", "ERROR")
        return False
    log("Installation and model verification completed successfully!", "SUCCESS")
    return True


def start_studio_services(args):
    """Starts Studio server and ComfyUI worker."""
    studio_pid, comfy_pid = get_running_pids()
    if studio_pid:
        log(f"Studio Server is already running with PID {studio_pid}.")
        return studio_pid

    cfg = load_studio_config()
    if args.port:
        cfg["port"] = int(args.port)
    if args.comfy_port:
        cfg["comfy_port"] = int(args.comfy_port)
    if args.host:
        cfg["host"] = args.host
    if args.reserve_vram is not None:
        cfg["reserve_vram_gb"] = float(args.reserve_vram)
    save_studio_config(cfg)
    update_extra_model_paths()

    py = get_python_exe()
    server_script = STUDIO_DIR / "server.py"

    log_file = open(LOGS_DIR / "studio.log", "a", encoding="utf-8")
    env = os.environ.copy()
    env["LTX_STUDIO_ROOT"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"

    log(f"Starting LTX Studio Server on http://{cfg['host']}:{cfg['port']}...")
    proc = subprocess.Popen(
        [py, str(server_script)],
        cwd=str(ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True if platform.system() != "Windows" else False
    )

    (STATE_DIR / "studio.pid").write_text(str(proc.pid))
    time.sleep(2.5)

    if not is_pid_alive(proc.pid):
        log("Studio server failed to start! Last lines of logs/studio.log:", "ERROR")
        try:
            with open(LOGS_DIR / "studio.log", "r", encoding="utf-8") as f:
                lines = f.readlines()[-20:]
                print("".join(lines))
        except Exception:
            pass
        return None

    log(f"Studio Server active with PID {proc.pid}.", "SUCCESS")
    return proc.pid


def wait_for_health(host, port, timeout=15):
    """Polls the studio server until it responds."""
    import urllib.request
    t0 = time.time()
    url = f"http://127.0.0.1:{port}/api/status"
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    parser = argparse.ArgumentParser(
        description="LTX 2.5 Video Enhancement Studio Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--daemon", action="store_true", help="Launch Studio in the background and return immediately")
    parser.add_argument("--install", action="store_true", help="Force running installer before launching")
    parser.add_argument("--install-only", action="store_true", help="Run full installation and model download, then exit")
    parser.add_argument("--run-only", action="store_true", help="Start services directly without running installer checks")
    parser.add_argument("--skip-models", action="store_true", help="Skip model weights download in installer")
    parser.add_argument("--verify-only", action="store_true", help="Recheck installed weights, worker and nodes without downloads or dependency installation")
    parser.add_argument("--no-gguf", action="store_true", default=True, help="Skip downloading optional 15.7 GB GGUF model")
    parser.add_argument("--with-gguf", action="store_false", dest="no_gguf", help="Download optional 15.7 GB GGUF model")
    parser.add_argument("--port", type=int, default=None, help="Studio Web UI port (default: 7860)")
    parser.add_argument("--comfy-port", type=int, default=None, help="ComfyUI internal port (default: 8189)")
    parser.add_argument("--host", type=str, default=None, help="Listen host (default: 0.0.0.0)")
    parser.add_argument("--reserve-vram", type=float, default=None, help="VRAM reserve in GB (default: 2.0)")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face API token for gated weights")
    parser.add_argument("--status", action="store_true", help="Display status and exit")
    parser.add_argument("--stop", action="store_true", help="Stop running services and exit")
    parser.add_argument("--restart", action="store_true", help="Restart Studio services")

    args = parser.parse_args()
    if args.verify_only:
        args.install_only = True

    # Subcommands
    if args.status:
        cmd_status()
        return

    if args.stop:
        cmd_stop()
        return

    if args.restart:
        cmd_stop()
        time.sleep(1.5)

    print(BANNER)

    # Pre-flight
    gpu = check_gpu()
    if gpu["present"]:
        log(f"Detected GPU: {gpu['name']} ({gpu['vram_mb']:.0f} MB VRAM)")
    else:
        log("No NVIDIA GPU detected. LTX Studio runs best with CUDA.", "WARN")

    free_gb = get_disk_free_gb(ROOT)
    log(f"Disk space available: {free_gb:.1f} GB")
    if free_gb < 40:
        log("Available disk space is under 40 GB. At least 100 GB is recommended for models and cache.", "WARN")

    # Installer phase
    is_installed = check_is_installed()
    needs_install = args.install or args.install_only or (not is_installed and not args.run_only)

    if needs_install:
        ok = run_installer(args)
        if not ok and args.install_only:
            sys.exit(1)
        if args.install_only:
            log("Install-only complete. You can now launch with: python launcher.py", "SUCCESS")
            return

    # Start services
    pid = start_studio_services(args)
    if not pid:
        sys.exit(1)

    cfg = load_studio_config()
    healthy = wait_for_health(cfg["host"], cfg["port"], timeout=15)
    if healthy:
        log("Health check passed! Studio Web UI is fully responsive.", "SUCCESS")
    else:
        log("Server process is active; awaiting worker initialization.", "WARN")

    # Output connection info
    port = cfg["port"]
    h3_ok, h3_busy = check_h3_status(cfg["h3_url"])
    print("\n" + "=" * 80)
    print(" 🚀  LTX VIDEO ENHANCEMENT STUDIO IS LIVE")
    print("=" * 80)
    print(f" 🌐  Direct Browser URL:   http://localhost:{port}  or  http://127.0.0.1:{port}")
    print(f" 🌐  Network IP URL:       http://0.0.0.0:{port}")
    print(f" 🔐  SSH Tunnel Command:   ssh -L {port}:127.0.0.1:{port} <user>@<remote-ip>")
    print(f" ⚙️  ComfyUI Backend:      http://127.0.0.1:{cfg['comfy_port']}")
    if h3_ok:
        print(f" 🤝  MiniMax H3 Queue:     CONNECTED ({'BUSY' if h3_busy else 'IDLE'}) — automatic queue active")
    print("-" * 80)
    print(" Useful commands:")
    print("   Check live status:      python launcher.py --status")
    print("   Stop all services:      python launcher.py --stop")
    print("   Server log:             tail -f logs/studio.log")
    print("=" * 80 + "\n")

    if args.daemon:
        log("Daemon mode active. Background services running. Exiting launcher.", "SUCCESS")
        return

    # Foreground supervision loop
    log("Running in supervision mode. Press Ctrl+C at any time to cleanly stop all services.\n")

    def handle_sigint(signum, frame):
        print("\n")
        log("Shutdown requested by user (Ctrl+C)...")
        cmd_stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    try:
        while True:
            time.sleep(2.0)
            studio_pid, _ = get_running_pids()
            if not studio_pid:
                log("Studio server process terminated unexpectedly! Check logs/studio.log", "ERROR")
                break
    except KeyboardInterrupt:
        handle_sigint(None, None)


if __name__ == "__main__":
    main()
