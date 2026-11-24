"""Shared paths, config and small helpers for the LTX Studio backend."""
import json, os, re, secrets, shutil, subprocess, threading, time, uuid
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get("LTX_STUDIO_ROOT", PROJECT)).resolve()
STATIC = Path(__file__).resolve().parent / "static"
REDETAIL = ROOT / "vendor" / "redetail"
if not REDETAIL.exists():  # development / tests: fall back to a sibling checkout
    for cand in (PROJECT / "vendor" / "redetail", PROJECT.parent / "redetail"):
        if cand.exists():
            REDETAIL = cand
            break

DIRS = {k: ROOT / k for k in ("inputs", "outputs", "jobs", "logs", "state", "uploads", "presets", "cache", "config", "models", "comfy_io")}
for d in DIRS.values():
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

DEFAULT_CONFIG = {
    "host": "0.0.0.0", "port": 7860, "comfy_port": 8189,
    "reserve_vram_gb": 2.0, "attention": "ck-int8", "cache_mode": "ram",
    "coordinate_h3": True, "h3_url": "http://127.0.0.1:8188",
    "release_vram_idle_s": 90, "auto_resume_jobs": True, "comfy_extra_args": [],
    "alloc_conf": "expandable_segments:True",
}


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    p = DIRS["config"] / "studio.json"
    if p.exists():
        try:
            cfg.update(json.loads(p.read_text()))
        except Exception:
            pass
    return cfg


def save_config(cfg):
    p = DIRS["config"] / "studio.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=2))
    os.replace(tmp, p)


def atomic_write_json(path: Path, obj):
    tmp = Path(str(path) + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


SAFE_ID = re.compile(r"^[A-Za-z0-9_.\-]{1,96}$")


def safe_id(s: str) -> str:
    if not SAFE_ID.match(s or "") or ".." in s:
        raise ValueError("invalid id")
    return s


def new_id(prefix):
    return f"{prefix}_{int(time.time()):x}{uuid.uuid4().hex[:5]}"


def internal_token():
    p = DIRS["state"] / "internal.token"
    if not p.exists():
        p.write_text(secrets.token_hex(24))
        os.chmod(p, 0o600)
    return p.read_text().strip()


def run(cmd, timeout=3600, check=False):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout)
    if check and r.returncode:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr[-400:]}")
    return r


def ram_limits():
    """(total_gb, available_gb) honouring a cgroup memory limit if one is set."""
    import psutil
    vm = psutil.virtual_memory()
    total, avail = vm.total, vm.available
    for f in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(f).read_text().strip()
            if v.isdigit() and int(v) < total:
                total = int(v)
                used = 0
                for u in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
                    try:
                        used = int(Path(u).read_text().strip())
                        break
                    except Exception:
                        pass
                avail = min(avail, max(0, total - used))
        except Exception:
            pass
    return total / 1e9, avail / 1e9


def fmt_eta(sec):
    if sec is None:
        return "-"
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")
