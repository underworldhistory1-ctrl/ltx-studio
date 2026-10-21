#!/usr/bin/env python3
"""LTX Video Enhancement Studio - automatic installer.

Safe to re-run: every phase is idempotent, downloads resume (block-level) and a failed/blocked
file never aborts the rest of the install.  Progress is written to <root>/state/install.json
(atomically, ~2 Hz) and streamed to the Studio UI over SSE, so a browser refresh loses nothing.

Nothing outside <root> is modified.  The existing H3 stack (/output/ComfyUI, port 8188) is never
touched; the base Python environment is only *read* (its site-packages are exposed to our private
venv through a .pth file, so torch / comfy-kitchen are not re-downloaded or changed).
"""
import argparse, json, os, shutil, signal, socket, subprocess, sys, threading, time, traceback, hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
MANIFEST = json.loads((HERE / "manifest.json").read_text())

# ---------------------------------------------------------------- IPv4 only (this host has a broken v6 route)
import urllib3.util.connection as _u3c
_u3c.allowed_gai_family = lambda: socket.AF_INET
import requests

PHASES = [  # id, label, weight (models phase weight is distributed by bytes)
    ("preflight", "Pre-flight checks (disk, GPU, tools)", 1),
    ("system", "System tools (ffmpeg, git)", 1),
    ("repos", "ComfyUI + ReDetail source", 3),
    ("venv", "Private Python environment", 2),
    ("deps", "Python dependencies", 6),
    ("nodes", "Custom nodes (LTXVideo, GGUF, cond-cache)", 3),
    ("models", "Model weights", 72),
    ("config", "Paths and configuration", 1),
    ("verify", "Integrity verification", 3),
    ("services", "Start ComfyUI worker", 3),
    ("smoke", "Smoke test (nodes + models visible)", 5),
]


def now():
    return time.time()


class State:
    """Thread-safe progress document persisted atomically."""

    def __init__(self, root: Path):
        self.root = root
        self.path = root / "state" / "install.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.dirty = False
        self.doc = {
            "status": "running", "started": now(), "updated": now(), "pid": os.getpid(),
            "version": MANIFEST["version"],
            "overall": {"pct": 0.0, "eta_s": None, "bytes_total": 0, "bytes_done": 0},
            "phases": [{"id": i, "label": l, "weight": w, "status": "pending", "pct": 0, "detail": ""} for i, l, w in PHASES],
            "current": None, "files": {}, "log": [], "actions": [], "error": None,
            "paths": {"root": str(root)},
        }
        self._stop = False
        self._speed_hist = []
        self.flush()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stop:
            time.sleep(0.5)
            with self.lock:
                self._recalc()
                if self.dirty:
                    self.flush()

    def _recalc(self):
        d = self.doc
        total = sum(f["size"] for f in d["files"].values()) or 0
        done = sum(min(f["done"], f["size"]) for f in d["files"].values())
        d["overall"]["bytes_total"], d["overall"]["bytes_done"] = total, done
        pct = 0.0
        for p in d["phases"]:
            if p["id"] == "models":
                frac = (done / total) if total else (1.0 if p["status"] == "done" else 0.0)
                if p["status"] == "done":
                    frac = 1.0
                p["pct"] = round(frac * 100, 1)
                pct += p["weight"] * frac
            else:
                frac = 1.0 if p["status"] in ("done", "skipped") else (p["pct"] / 100.0 if p["status"] == "running" else 0.0)
                pct += p["weight"] * frac
        d["overall"]["pct"] = round(min(pct, 100.0), 2)
        cur = d.get("current")
        if cur:
            t = now()
            self._speed_hist.append((t, cur["done"] + sum(min(f["done"], f["size"]) for k, f in d["files"].items() if k != cur.get("id"))))
            self._speed_hist = [x for x in self._speed_hist if t - x[0] <= 8]
            if len(self._speed_hist) > 2:
                (t0, b0), (t1, b1) = self._speed_hist[0], self._speed_hist[-1]
                bps = max(0.0, (b1 - b0) / max(t1 - t0, 1e-6))
                cur["speed_bps"] = bps
                cur["eta_s"] = round((cur["total"] - cur["done"]) / bps) if bps > 1000 else None
                rem_all = max(0, total - done)
                d["overall"]["eta_s"] = round(rem_all / bps) if bps > 1000 else None
        else:
            d["overall"]["eta_s"] = None
        d["updated"] = now()

    def flush(self):
        with self.lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.doc, indent=1))
            os.replace(tmp, self.path)
            self.dirty = False

    def log(self, msg, level="info"):
        with self.lock:
            line = {"t": round(now(), 1), "level": level, "msg": str(msg)[:400]}
            self.doc["log"] = (self.doc["log"] + [line])[-80:]
            self.dirty = True
        print(f"[{level}] {msg}", flush=True)

    def phase(self, pid, status=None, pct=None, detail=None):
        with self.lock:
            for p in self.doc["phases"]:
                if p["id"] == pid:
                    if status:
                        p["status"] = status
                        if status == "done":
                            p["pct"] = 100
                    if pct is not None:
                        p["pct"] = pct
                    if detail is not None:
                        p["detail"] = detail
            self.dirty = True

    def set(self, **kw):
        with self.lock:
            self.doc.update(kw)
            self.dirty = True

    def file(self, fid, **kw):
        with self.lock:
            self.doc["files"].setdefault(fid, {"done": 0, "size": 0, "status": "pending"}).update(kw)
            self.dirty = True

    def action(self, aid, message, **extra):
        with self.lock:
            self.doc["actions"] = [a for a in self.doc["actions"] if a["id"] != aid] + [{"id": aid, "message": message, **extra}]
            self.dirty = True

    def clear_action(self, aid):
        with self.lock:
            self.doc["actions"] = [a for a in self.doc["actions"] if a["id"] != aid]
            self.dirty = True

    def close(self):
        self._stop = True
        with self.lock:
            self._recalc()
            self.flush()


# ---------------------------------------------------------------- shell helpers
def run(cmd, st: State = None, cwd=None, env=None, timeout=3600, phase=None, check=True):
    """Run a command, streaming its last output line into the phase detail."""
    if st:
        st.log("$ " + " ".join(map(str, cmd))[:300])
    p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    t0 = now()
    for line in p.stdout:
        line = line.rstrip()
        if not line:
            continue
        tail = (tail + [line])[-12:]
        if st and phase:
            st.phase(phase, detail=line[:160])
        if now() - t0 > timeout:
            p.kill()
            raise RuntimeError("timeout: " + " ".join(map(str, cmd))[:100])
    rc = p.wait()
    if check and rc != 0:
        raise RuntimeError(f"command failed ({rc}): {' '.join(map(str, cmd))[:120]}\n" + "\n".join(tail[-6:]))
    return rc, "\n".join(tail)


# ---------------------------------------------------------------- downloader
BLOCK = 32 * 1024 * 1024
WORKERS = 6


class GatedError(Exception):
    pass


def source_url(src, mirror_base=None):
    t = src["type"]
    if t == "modelscope":
        from urllib.parse import quote
        return f"https://www.modelscope.cn/api/v1/models/{src['repo']}/repo?Revision=master&FilePath={quote(src['path'], safe='')}"
    if t == "hf":
        return f"https://huggingface.co/{src['repo']}/resolve/main/{src['path']}"
    if t == "hfmirror":
        return f"https://hf-mirror.com/{src['repo']}/resolve/main/{src['path']}"
    raise ValueError(t)


def http_get(url, headers, token=None, stream=True, timeout=30):
    """GET with manual redirects; the Authorization header is NEVER forwarded to another host."""
    from urllib.parse import urlparse, urljoin
    host0 = urlparse(url).netloc
    h = dict(headers)
    h["User-Agent"] = "Mozilla/5.0 ltx-studio-installer"
    for _ in range(8):
        hh = dict(h)
        if token and urlparse(url).netloc == host0:
            hh["Authorization"] = f"Bearer {token}"
        r = requests.get(url, headers=hh, stream=stream, allow_redirects=False, timeout=timeout)
        if r.status_code in (301, 302, 303, 307, 308):
            url = urljoin(url, r.headers["Location"])
            r.close()
            continue
        return r
    raise RuntimeError("too many redirects")


def probe_size(url, token):
    r = http_get(url, {"Range": "bytes=0-0"}, token)
    try:
        if r.status_code in (401, 403):
            raise GatedError(f"HTTP {r.status_code}")
        if r.status_code == 206:
            cr = r.headers.get("Content-Range", "")
            return int(cr.rsplit("/", 1)[1]), True, r.url
        if r.status_code == 200:
            return int(r.headers.get("Content-Length", 0)), False, r.url
        raise RuntimeError(f"HTTP {r.status_code}")
    finally:
        r.close()


def verify_file(path: Path, kind: str, size: int):
    if path.stat().st_size != size:
        return False, f"size {path.stat().st_size} != {size}"
    try:
        with open(path, "rb") as f:
            if kind == "safetensors":
                n = int.from_bytes(f.read(8), "little")
                if n <= 0 or n > 100_000_000:
                    return False, "bad safetensors header length"
                hdr = json.loads(f.read(n))
                end = 0
                for k, v in hdr.items():
                    if k != "__metadata__":
                        end = max(end, v["data_offsets"][1])
                if 8 + n + end != size:
                    return False, f"safetensors tensor data ends at {8 + n + end}, file is {size}"
            elif kind == "gguf":
                if f.read(4) != b"GGUF":
                    return False, "bad GGUF magic"
    except Exception as e:
        return False, f"unreadable: {e}"
    return True, "ok"


class Download:
    def __init__(self, st: State, item, token):
        self.st, self.item, self.token = st, item, token
        self.dest = Path(item["_dest"])
        self.part = Path(str(self.dest) + ".part")
        self.meta = Path(str(self.dest) + ".part.json")
        self.size = item["size"]
        self.lock = threading.Lock()
        self.done_blocks = set()
        self.inflight = {}
        self.err = None

    def bytes_done(self):
        with self.lock:
            return min(self.size, len(self.done_blocks) * BLOCK + sum(self.inflight.values()))

    def _load_meta(self):
        try:
            m = json.loads(self.meta.read_text())
            if m.get("size") == self.size and self.part.exists() and self.part.stat().st_size == self.size:
                self.done_blocks = set(m.get("done", []))
                return
        except Exception:
            pass
        self.done_blocks = set()

    def _save_meta(self):
        with self.lock:
            snap = sorted(self.done_blocks)
        tmp = self.meta.with_suffix(".tmp")
        tmp.write_text(json.dumps({"size": self.size, "done": snap}))
        os.replace(tmp, self.meta)

    def run_source(self, src, attempt):
        url = source_url(src)
        tok = self.token if src["type"] == "hf" else None
        if src.get("gated") and not tok:
            raise GatedError("gated source, no token")
        total, ranges, final = probe_size(url, tok)
        if total != self.size:
            raise RuntimeError(f"source reports {total} bytes, expected {self.size}")
        self.dest.parent.mkdir(parents=True, exist_ok=True)
        self._load_meta()
        if not self.part.exists() or self.part.stat().st_size != self.size:
            with open(self.part, "wb") as f:
                f.truncate(self.size)
            self.done_blocks = set()
        nblocks = (self.size + BLOCK - 1) // BLOCK
        todo = [b for b in range(nblocks) if b not in self.done_blocks]
        st = self.st
        st.file(self.item["id"], source=src["type"], status="downloading", message="")
        cur = {"id": self.item["id"], "file": self.item["name"], "label": self.item["label"], "total": self.size,
               "done": self.bytes_done(), "pct": 0, "speed_bps": 0, "eta_s": None, "source": src["type"], "attempt": attempt,
               "mode": f"{'ranged x' + str(WORKERS) if ranges else 'single stream'}"}
        with st.lock:
            st.doc["current"] = cur
        qlock = threading.Lock()
        queue = list(todo)
        stop = threading.Event()
        fd = os.open(self.part, os.O_RDWR)

        def worker():
            while not stop.is_set():
                with qlock:
                    if not queue:
                        return
                    b = queue.pop(0)
                lo = b * BLOCK if ranges else 0
                hi = (min(self.size, lo + BLOCK) - 1) if ranges else self.size - 1
                for tries in range(6):
                    try:
                        r = http_get(url, {"Range": f"bytes={lo}-{hi}"} if ranges else {}, tok, timeout=40)
                        if r.status_code in (401, 403):
                            raise GatedError(f"HTTP {r.status_code}")
                        if ranges and r.status_code != 206:
                            raise RuntimeError(f"range not honoured: HTTP {r.status_code}")
                        if not ranges and r.status_code != 200:
                            raise RuntimeError(f"HTTP {r.status_code}")
                        pos = lo
                        with self.lock:
                            self.inflight[b] = 0
                        for chunk in r.iter_content(1 << 20):
                            if stop.is_set():
                                r.close()
                                return
                            os.pwrite(fd, chunk, pos)
                            pos += len(chunk)
                            with self.lock:
                                self.inflight[b] = pos - lo
                        r.close()
                        if pos - 1 != hi:
                            raise RuntimeError(f"short block {b}: {pos - lo}/{hi - lo + 1}")
                        with self.lock:
                            self.inflight.pop(b, None)
                            self.done_blocks.add(b)
                        break
                    except GatedError as e:
                        self.err = e
                        stop.set()
                        return
                    except Exception as e:
                        with self.lock:
                            self.inflight.pop(b, None)
                        if tries == 5:
                            self.err = e
                            stop.set()
                            return
                        time.sleep(min(30, 2 ** tries))
                if not ranges:  # single stream covers everything in "block 0"
                    with qlock:
                        queue.clear()
                    with self.lock:
                        self.done_blocks.update(range(nblocks))

        # without Range support the whole file is one block: emulate with a single worker
        if not ranges:
            queue[:] = [0]
            self.done_blocks = set()
        threads = [threading.Thread(target=worker, daemon=True) for _ in range(WORKERS if ranges else 1)]
        for t in threads:
            t.start()
        last_meta = now()
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
            d = self.bytes_done()
            with st.lock:
                cur["done"] = d
                cur["pct"] = round(d * 100.0 / self.size, 2)
                st.doc["files"][self.item["id"]]["done"] = d
                st.dirty = True
            if now() - last_meta > 3:
                self._save_meta()
                last_meta = now()
        os.close(fd)
        self._save_meta()
        if self.err:
            e, self.err = self.err, None
            raise e
        ok, why = verify_file(self.part, self.item["kind"], self.size)
        if not ok:
            self.part.unlink(missing_ok=True)
            self.meta.unlink(missing_ok=True)
            raise RuntimeError("verification failed after download: " + why)
        os.replace(self.part, self.dest)
        self.meta.unlink(missing_ok=True)


def download_item(st: State, item, token):
    fid = item["id"]
    dest = Path(item["_dest"])
    if dest.exists():
        ok, why = verify_file(dest, item["kind"], item["size"])
        if ok:
            st.file(fid, done=item["size"], status="done", message="already present and verified")
            return True
        st.log(f"{item['name']}: existing file invalid ({why}); re-downloading", "warn")
        dest.rename(str(dest) + ".invalid")
    dl = Download(st, item, token)
    st.file(fid, label=item["label"], name=item["name"], size=item["size"], done=dl.bytes_done() if dl.part.exists() else 0, status="pending", group=item["group"], required=item["required"])
    blocked = None
    for cycle in range(3):
        for src in item["sources"]:
            try:
                st.log(f"{item['name']}: trying {src['type']} (cycle {cycle + 1})")
                dl.run_source(src, cycle + 1)
                st.file(fid, done=item["size"], status="done", message=f"from {src['type']}")
                with st.lock:
                    st.doc["current"] = None
                return True
            except GatedError as e:
                blocked = f"{src['type']}: {e}"
                st.log(f"{item['name']}: {src['type']} requires licence/token ({e})", "warn")
            except Exception as e:
                st.log(f"{item['name']}: {src['type']} failed: {e}", "warn")
        if cycle < 2:
            wait = 15 * (cycle + 1)
            st.file(fid, status="retrying", message=f"all sources failed; retrying in {wait}s")
            time.sleep(wait)
    with st.lock:
        st.doc["current"] = None
    if blocked and not token:
        st.file(fid, status="needs_token", message="Download needs a Hugging Face token with the licence accepted. " + blocked)
        st.action("hf_token", "Some models need a Hugging Face token (accept the LTX-2 licence on the model page, then paste a read token in Settings > Install).", files=[fid])
    else:
        st.file(fid, status="failed", message="all sources failed; re-run the installer to resume")
    return False


# ---------------------------------------------------------------- phases
def read_token(root: Path):
    t = os.environ.get("HF_TOKEN", "").strip()
    f = root / "config" / "hf_token"
    if not t and f.exists():
        t = f.read_text().strip()
    return t or None


def phase_preflight(st, root, args):
    st.phase("preflight", "running", 10)
    root.mkdir(parents=True, exist_ok=True)
    st.log(f"install root: {root}")
    if str(root).startswith(("/output/ComfyUI", "/output/h3-")) or root.resolve() == Path("/output"):
        raise RuntimeError("refusing to install inside an existing H3 directory")
    free = shutil.disk_usage(root).free / 1e9
    need_models = sum(m["size"] for m in MANIFEST["models"] if (args.with_gguf or m["group"] != "gguf")) / 1e9
    st.log(f"free disk at root: {free:.0f} GB; models need {need_models:.0f} GB (+ scratch)")
    if free < max(MANIFEST["min_free_disk_gb"], need_models * 1.5):
        raise RuntimeError(f"not enough disk space: {free:.0f} GB free, need ~{max(MANIFEST['min_free_disk_gb'], need_models * 1.5):.0f} GB")
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True).strip()
        st.log("GPU: " + out)
        st.set(gpu=out)
    except Exception:
        st.log("nvidia-smi not found - rendering will not work without an NVIDIA GPU", "warn")
    for tool in ("git",):
        if not shutil.which(tool):
            raise RuntimeError(f"{tool} is required")
    st.phase("preflight", "done")


def phase_system(st, root, args):
    st.phase("system", "running", 30)
    missing = [t for t in ("ffmpeg", "ffprobe") if not shutil.which(t)]
    if missing:
        st.log(f"{missing} missing; trying apt", "warn")
        try:
            run(["apt-get", "install", "-y", "ffmpeg"], st, phase="system", timeout=900)
        except Exception as e:
            raise RuntimeError("ffmpeg/ffprobe are required and could not be installed: " + str(e))
    out = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]
    st.log(out)
    # the colour pipeline needs OpenCV + numpy (read from the base env, no install)
    st.phase("system", "done")


def git_sync(st, url, dest: Path, commit=None, branch=None, fallback=None, phase="repos"):
    if (dest / ".git").exists():
        st.log(f"{dest.name}: present")
        if commit:
            r = subprocess.run(["git", "-C", dest, "rev-parse", "HEAD"], capture_output=True, text=True)
            if r.stdout.strip() == commit:
                return
        else:
            return
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        for attempt in range(5):
            cmd = (["git", "clone", "--filter=blob:none", "--no-checkout", url, dest] if commit
                   else ["git", "clone", "--depth", "1", "-b", branch, url, dest])
            rc, _ = run(cmd, st, phase=phase, check=False, timeout=1800)
            if rc == 0:
                break
            shutil.rmtree(dest, ignore_errors=True)
            time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"could not clone {url}")
    if commit:
        for c in [commit] + ([fallback] if fallback else []):
            rc, _ = run(["git", "-C", dest, "checkout", "--detach", c], st, check=False)
            if rc == 0:
                st.log(f"{dest.name} @ {c[:10]}")
                return
            run(["git", "-C", dest, "fetch", "--depth", "1", "origin", c], st, check=False)
            rc, _ = run(["git", "-C", dest, "checkout", "--detach", c], st, check=False)
            if rc == 0:
                st.log(f"{dest.name} @ {c[:10]}")
                return
        raise RuntimeError(f"cannot check out {commit} in {dest}")


def phase_repos(st, root, args):
    st.phase("repos", "running", 5)
    R = MANIFEST["repos"]
    git_sync(st, R["comfyui"]["url"], root / "ComfyUI", commit=R["comfyui"]["commit"], fallback=R["comfyui"].get("fallback_commit"))
    st.phase("repos", pct=60)
    git_sync(st, R["redetail"]["url"], root / "vendor" / "redetail", commit=R["redetail"]["commit"])
    st.phase("repos", "done")


def venv_python(root):
    return root / "venv" / "bin" / "python"


def pip(st, root, args, phase):
    py = venv_python(root)
    cons = root / "config" / "constraints.txt"
    base_args = list(args) + (["-c", str(cons)] if cons.exists() else [])
    rc, out = run([py, "-m", "pip", "install", "--no-input", "--disable-pip-version-check", *base_args], st, phase=phase, check=False, timeout=3600)
    if rc != 0 and shutil.which("uv"):
        st.log("pip failed, retrying with uv", "warn")
        rc, out = run(["uv", "pip", "install", "--python", py, *base_args], st, phase=phase, check=False, timeout=3600)
    if rc != 0:
        raise RuntimeError("pip install failed: " + out[-400:])


def phase_venv(st, root, args):
    st.phase("venv", "running", 10)
    py = venv_python(root)
    # Determine the host/system base python that has PyTorch and CUDA
    base_candidates = [Path(sys.executable), Path("/opt/venv/bin/python"), Path("/workspace/venv/bin/python")]
    base_sp = None
    base_py = Path(sys.executable)
    for cand in base_candidates:
        if cand.exists() and cand.resolve() != py.resolve():
            try:
                sp = subprocess.check_output([str(cand), "-c", "import site;print(site.getsitepackages()[0])"], text=True).strip()
                if Path(sp).exists() and (Path(sp) / "torch").exists():
                    base_sp = sp
                    base_py = cand
                    break
            except Exception:
                pass
    if not base_sp:
        try:
            base_sp = subprocess.check_output([str(base_py), "-c", "import site;print(site.getsitepackages()[0])"], text=True).strip()
        except Exception:
            base_sp = "/opt/venv/lib/python3.12/site-packages"

    if not py.exists():
        rc, _ = run([str(base_py), "-m", "venv", root / "venv"], st, check=False)
        if rc != 0 or not py.exists():
            shutil.rmtree(root / "venv", ignore_errors=True)
            run(["uv", "venv", "--seed", "--python", str(base_py), root / "venv"], st)
    sp = subprocess.check_output([py, "-c", "import site;print(site.getsitepackages()[0])"], text=True).strip()
    (Path(sp) / "zz_ltx_base_env.pth").write_text(base_sp + "\n")  # exposes torch & co. read-only, AFTER our own packages
    # constraints: keep the heavy base packages exactly as they are
    cons = root / "config" / "constraints.txt"
    cons.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for pkg in ("torch", "torchvision", "torchaudio", "numpy", "triton", "opencv-python-headless"):
        r = subprocess.run([str(base_py), "-m", "pip", "show", pkg], capture_output=True, text=True)
        for l in r.stdout.splitlines():
            if l.startswith("Version:"):
                lines.append(f"{pkg}=={l.split(':', 1)[1].strip()}")
    cons.write_text("\n".join(lines) + "\n")
    out = subprocess.check_output([py, "-c", "import torch,comfy_kitchen;print(torch.__version__, torch.cuda.is_available(), comfy_kitchen.__version__ if hasattr(comfy_kitchen,'__version__') else '')"], text=True).strip()
    st.log("venv sees: " + out)
    if " True" not in out:
        st.log("torch.cuda is not available in the venv", "warn")
    st.phase("venv", "done")


def phase_deps(st, root, args):
    st.phase("deps", "running", 5)
    pip(st, root, ["-r", str(root / "ComfyUI" / "requirements.txt")], "deps")
    st.phase("deps", pct=70)
    # comfy-kitchen >= 0.2.26 is mandatory for int8 convrot (the base env has 0.2.35)
    r = subprocess.run([str(venv_python(root)), "-c", "import comfy_kitchen as c;from importlib.metadata import version;print(version('comfy-kitchen'))"], capture_output=True, text=True)
    ver = r.stdout.strip()
    st.log(f"comfy-kitchen {ver}")
    try:
        ok = tuple(int(x) for x in ver.split(".")[:3]) >= (0, 2, 26)
    except Exception:
        ok = False
    if not ok:
        pip(st, root, ["comfy-kitchen>=0.2.26"], "deps")
    st.phase("deps", "done")


def phase_nodes(st, root, args):
    st.phase("nodes", "running", 5)
    cn = root / "ComfyUI" / "custom_nodes"
    R = MANIFEST["repos"]
    git_sync(st, R["ltxvideo"]["url"], cn / "ComfyUI-LTXVideo", branch=R["ltxvideo"]["branch"], phase="nodes")
    req = cn / "ComfyUI-LTXVideo" / "requirements.txt"
    if req.exists():
        pip(st, root, ["-r", str(req)], "nodes")
    st.phase("nodes", pct=55)
    git_sync(st, R["gguf"]["url"], cn / "ComfyUI-GGUF", branch=R["gguf"]["branch"], phase="nodes")
    req = cn / "ComfyUI-GGUF" / "requirements.txt"
    if req.exists():
        pip(st, root, ["-r", str(req)], "nodes")
    st.phase("nodes", pct=80)
    src = root / "vendor" / "redetail" / "tools" / "comfyui_cond_cache"
    dst = cn / "comfyui_cond_cache"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    st.log("copied ReDetail cond-cache node")
    shutil.copytree(PROJECT / "comfy_nodes" / "studio_shared_gpu", cn / "studio_shared_gpu", dirs_exist_ok=True)
    st.log("copied optional shared-GPU queue admission node")
    st.phase("nodes", "done")


def phase_models(st, root, args):
    st.phase("models", "running")
    token = read_token(root)
    st.log("Hugging Face token: " + ("provided" if token else "none (ModelScope mirrors are ungated)"))
    mdir = root / "models"
    items = []
    for m in MANIFEST["models"]:
        if m["group"] == "gguf" and not args.with_gguf:
            continue
        m = dict(m)
        m["_dest"] = str(mdir / m["dir"] / m["name"])
        items.append(m)
        dest = Path(m["_dest"])
        part = Path(m["_dest"] + ".part")
        done = m["size"] if dest.exists() and dest.stat().st_size == m["size"] else 0
        st.file(m["id"], label=m["label"], name=m["name"], size=m["size"], done=done, status="done" if done else "pending", group=m["group"], required=m["required"])
    st.clear_action("hf_token")
    # core first (smallest first so the UI becomes useful early), GGUF last
    items.sort(key=lambda m: (m["group"] == "gguf", m["size"]))
    failures = []
    for m in items:
        if not download_item(st, m, token):
            failures.append(m)
    with st.lock:
        st.doc["current"] = None
    req_fail = [m for m in failures if m["required"]]
    if failures:
        st.phase("models", "failed" if req_fail else "done", detail=f"{len(failures)} file(s) not downloaded: " + ", ".join(f['name'] for f in failures))
        st.log(f"{len(failures)} model file(s) not downloaded", "warn")
    else:
        st.phase("models", "done", detail="all model files present")
    return failures


def phase_config(st, root, args):
    st.phase("config", "running", 20)
    cfg = root / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    for d in ("inputs", "outputs", "jobs", "logs", "comfy_io/input", "comfy_io/output", "comfy_io/temp", "state", "uploads", "presets", "cache"):
        (root / d).mkdir(parents=True, exist_ok=True)
    (cfg / "extra_model_paths.yaml").write_text(
        "ltx_studio:\n"
        f"    base_path: {root.as_posix()}/models\n"
        "    diffusion_models: diffusion_models\n"
        "    text_encoders: text_encoders\n"
        "    clip: text_encoders\n"
        "    vae: vae\n"
        "    loras: loras\n"
        "    unet: unet\n")
    if not (cfg / "studio.json").exists():
        (cfg / "studio.json").write_text(json.dumps({
            "host": "127.0.0.1", "port": MANIFEST["ports"]["studio"], "comfy_port": MANIFEST["ports"]["comfy"],
            "reserve_vram_gb": 2.0, "attention": "pytorch", "coordinate_h3": True, "h3_url": "http://127.0.0.1:8188",
            "release_vram_idle_s": 90, "auto_resume_jobs": True, "comfy_extra_args": []}, indent=2))
    st.phase("config", "done")


def phase_verify(st, root, args, failures):
    st.phase("verify", "running", 10)
    bad = []
    for m in MANIFEST["models"]:
        if m["group"] == "gguf" and not args.with_gguf:
            continue
        p = root / "models" / m["dir"] / m["name"]
        if not p.exists():
            continue
        if p.is_symlink():
            bad.append(f"{m['name']} is a symlink/alias; aliases are never allowed")
            continue
        ok, why = verify_file(p, m["kind"], m["size"])
        if not ok:
            bad.append(f"{m['name']}: {why}")
    if bad:
        for b in bad:
            st.log("verify: " + b, "error")
        raise RuntimeError("; ".join(bad))
    st.phase("verify", "done")


def studio_api(root, path, method="POST"):
    tok = (root / "state" / "internal.token")
    if not tok.exists():
        return None
    cfg = json.loads((root / "config" / "studio.json").read_text())
    try:
        r = requests.request(method, f"http://127.0.0.1:{cfg['port']}{path}", headers={"X-Internal-Token": tok.read_text().strip()}, timeout=20)
        return r.status_code, r.text[:300]
    except Exception as e:
        return None


def phase_services(st, root, args, models_ok):
    st.phase("services", "running", 10)
    dit = root / "models" / "diffusion_models" / "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
    if not models_ok or not dit.exists() or dit.stat().st_size < 10_000_000:
        st.phase("services", "skipped", detail="core models not yet downloaded - worker not started")
        return False
    res = studio_api(root, "/api/internal/comfy/start")
    st.log(f"studio -> comfy start: {res}")
    if not res or res[0] != 200:
        st.phase("services", "failed", detail="Studio server did not accept the start request")
        return False
    st.phase("services", "done")
    return True


def phase_smoke(st, root, args):
    st.phase("smoke", "running", 5)
    cfg = json.loads((root / "config" / "studio.json").read_text())
    base = f"http://127.0.0.1:{cfg['comfy_port']}"
    t0 = now()
    oi = None
    while now() - t0 < 420:
        try:
            if requests.get(base + "/system_stats", timeout=5).ok:
                oi = requests.get(base + "/object_info", timeout=60).json()
                break
        except Exception:
            pass
        st.phase("smoke", pct=min(60, int((now() - t0) / 7)), detail="waiting for ComfyUI to boot")
        time.sleep(4)
    if not oi:
        st.phase("smoke", "failed", detail="ComfyUI did not answer; see logs/comfy.log")
        return False
    need = ["LTXAddVideoICLoRAGuide", "LTXICLoRALoaderModelOnly", "EmptyLTXVLatentVideo", "VAEDecodeTiled",
            "LTXVTiledFusionSampler", "LTXVGetTilingSizes", "ReDetailLoadConditioning", "ReDetailSaveConditioning"]
    miss = [n for n in need if n not in oi]
    opt_miss = [n for n in ("UnetLoaderGGUF",) if n not in oi]
    def enum(cls, field):
        try:
            return oi[cls]["input"]["required"][field][0]
        except Exception:
            return []
    models_seen = {
        "unet": enum("UNETLoader", "unet_name"), "clip": enum("CLIPLoader", "clip_name"),
        "vae": enum("VAELoader", "vae_name"), "lora": enum("LoraLoaderModelOnly", "lora_name"),
    }
    for m in MANIFEST["models"]:
        if m["group"] == "gguf":
            continue
        key = {"diffusion_models": "unet", "text_encoders": "clip", "vae": "vae", "loras": "lora"}[m["dir"]]
        if (root / "models" / m["dir"] / m["name"]).exists() and m["name"] not in models_seen[key]:
            miss.append(f"model {m['name']} not visible to ComfyUI")
    if miss:
        st.log("smoke test missing: " + ", ".join(miss), "error")
        st.phase("smoke", "failed", detail="missing: " + ", ".join(miss))
        return False
    if opt_miss:
        st.log("optional nodes missing: " + ", ".join(opt_miss), "warn")
    st.log("smoke test: all LTX/ReDetail nodes and model files are visible to ComfyUI")
    st.phase("smoke", "done")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("LTX_STUDIO_ROOT", str(PROJECT)))
    ap.add_argument("--no-gguf", dest="with_gguf", action="store_false", help="skip the optional 15.7 GB GGUF model")
    ap.add_argument("--skip-models", action="store_true")
    ap.set_defaults(with_gguf=True)
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    lock = open(root / ".install.lock", "w")
    try:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another installer run is active", file=sys.stderr)
        sys.exit(3)
    st = State(root)
    st.log(f"LTX Studio installer v{MANIFEST['version']} -> {root}")
    status = "done"
    failures = []
    try:
        phase_preflight(st, root, args)
        phase_system(st, root, args)
        phase_config(st, root, args)  # dirs + studio.json first: the web server needs them
        phase_repos(st, root, args)
        phase_venv(st, root, args)
        phase_deps(st, root, args)
        phase_nodes(st, root, args)
        failures = [] if args.skip_models else phase_models(st, root, args)
        phase_verify(st, root, args, failures)
        core_ok = not [m for m in failures if m["required"] and m["group"] == "core" and m["id"] in ("dit_int8", "vae_video", "vae_audio")]
        started = phase_services(st, root, args, core_ok)
        smoke = phase_smoke(st, root, args) if started else False
        if args.skip_models:
            status = "ready_for_models"
        elif failures:
            status = "needs_action"
        elif not smoke:
            status = "failed"
    except Exception as e:
        status = "failed"
        st.log("INSTALL FAILED: " + str(e), "error")
        st.set(error=str(e) + "\n" + traceback.format_exc()[-800:])
        for p in st.doc["phases"]:
            if p["status"] == "running":
                p["status"] = "failed"
    st.set(status=status, finished=now(), current=None)
    st.log(f"installer finished: {status}")
    st.close()
    sys.exit(0 if status in ("done", "needs_action") else 1)


if __name__ == "__main__":
    main()
