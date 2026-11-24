"""ComfyUI worker supervision + an async client for its HTTP/WebSocket API."""
import asyncio, json, os, signal, subprocess, sys, threading, time, uuid
from pathlib import Path

import aiohttp

from common import ROOT, DIRS, load_config, atomic_write_json, read_json, ram_limits


class PromptRejected(Exception):
    def __init__(self, msg, detail=None):
        super().__init__(msg)
        self.detail = detail


# ------------------------------------------------------------------ process supervisor
class ComfyProcess:
    def __init__(self):
        self.proc = None
        self.pid = None
        self.desired = False
        self.restarts = []
        self.last_error = None
        self.lock = threading.Lock()
        self._caps = None
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        self.pidfile = DIRS["state"] / "comfy.pid"
        self._adopt()

    # --- helpers
    def _alive(self, pid):
        if not pid:
            return False
        try:
            os.kill(pid, 0)
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                return b"main.py" in f.read()
        except Exception:
            return False

    def _adopt(self):
        self.pidfile = DIRS["state"] / "comfy.pid"
        info = read_json(self.pidfile, {})
        if info.get("pid") and self._alive(info["pid"]):
            self.pid = info["pid"]
            self.desired = True

    @property
    def root(self):
        return ROOT / "ComfyUI"

    def python(self):
        p = ROOT / "venv" / "bin" / "python"
        return p if p.exists() else Path(sys.executable)

    def capabilities(self):
        if self._caps is not None:
            return self._caps
        cache = DIRS["state"] / "comfy_caps.json"
        sig = ""
        try:
            sig = (self.root / ".git" / "HEAD").read_text().strip()
        except Exception:
            pass
        c = read_json(cache, {})
        if c.get("sig") == sig and "flags" in c:
            self._caps = c["flags"]
            return self._caps
        try:
            out = subprocess.run([str(self.python()), "main.py", "--help"], cwd=self.root, capture_output=True, text=True, timeout=180).stdout
        except Exception:
            out = ""
        flags = [f for f in ("--cache-none", "--cache-ram", "--cache-lru", "--fp16-intermediates", "--use-pytorch-cross-attention",
                             "--use-ck-attention", "--reserve-vram", "--disable-smart-memory", "--extra-model-paths-config",
                             "--input-directory", "--output-directory", "--temp-directory", "--disable-auto-launch", "--disable-cuda-malloc")
                 if f in out]
        atomic_write_json(cache, {"sig": sig, "flags": flags})
        self._caps = flags
        return flags

    def build_cmd(self):
        cfg = load_config()
        caps = self.capabilities()
        cmd = [str(self.python()), "main.py", "--listen", "127.0.0.1", "--port", str(cfg["comfy_port"])]
        if "--extra-model-paths-config" in caps:
            cmd += ["--extra-model-paths-config", str(DIRS["config"] / "extra_model_paths.yaml")]
        for flag, d in (("--input-directory", "input"), ("--output-directory", "output"), ("--temp-directory", "temp")):
            if flag in caps:
                (DIRS["comfy_io"] / d).mkdir(parents=True, exist_ok=True)
                cmd += [flag, str(DIRS["comfy_io"] / d)]
        if "--reserve-vram" in caps:
            cmd += ["--reserve-vram", str(cfg["reserve_vram_gb"])]
        if "--fp16-intermediates" in caps:
            cmd += ["--fp16-intermediates"]
        att = cfg.get("attention", "pytorch")
        if att == "ck-int8" and "--use-ck-attention" in caps:
            cmd += ["--use-ck-attention"]
        elif "--use-pytorch-cross-attention" in caps:
            cmd += ["--use-pytorch-cross-attention"]
        cm = cfg.get("cache_mode", "ram")
        extra_args = [str(a) for a in cfg.get("comfy_extra_args", [])]
        if cm == "none" and "--cache-none" in caps:
            cmd += ["--cache-none"]
        elif cm == "ram" and "--cache-ram" in caps:
            if "--cache-ram" not in extra_args:
                total_gb, _ = ram_limits()
                if 0 < total_gb < 56:
                    cmd += ["--cache-ram", "6", "12"]
                else:
                    cmd += ["--cache-ram"]
        if "--disable-auto-launch" in caps:
            cmd += ["--disable-auto-launch"]
        cmd += extra_args
        return cmd

    # --- control
    def start(self):
        with self.lock:
            self.desired = True
            if self.running():
                return True
            if not (self.root / "main.py").exists():
                self.last_error = "ComfyUI is not installed yet"
                return False
            cfg = load_config()
            env = dict(os.environ)
            env["PYTORCH_CUDA_ALLOC_CONF"] = cfg.get("alloc_conf", "expandable_segments:True")
            env["PYTHONUNBUFFERED"] = "1"
            env.pop("PYTHONHOME", None)
            if os.name == "posix" and cfg.get("coordinate_h3", True):
                h3_url = (cfg.get("h3_url") or "http://127.0.0.1:8188").rstrip("/")
                env.setdefault("STUDIO_SHARED_GPU_PEER", h3_url)
                env.setdefault("STUDIO_SHARED_GPU_LOCK", "/tmp/studio_shared_gpu.lock")
            log = open(DIRS["logs"] / "comfy.log", "ab")
            cmd = self.build_cmd()
            log.write(f"\n=== {time.ctime()} start: {' '.join(cmd)}\n".encode())
            log.flush()
            self.proc = subprocess.Popen(cmd, cwd=self.root, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            self.pid = self.proc.pid
            atomic_write_json(self.pidfile, {"pid": self.pid, "started": time.time(), "port": cfg["comfy_port"]})
            self.restarts = [t for t in self.restarts if time.time() - t < 600] + [time.time()]
            return True

    def running(self):
        if self.proc is not None:
            return self.proc.poll() is None
        return self._alive(self.pid)

    def stop(self, timeout=40):
        with self.lock:
            self.desired = False
            pid = self.pid
            if not pid or not self.running():
                self.proc = None
                self.pid = None
                return
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except Exception:
                try:
                    os.kill(pid, signal.SIGTERM)
                except Exception:
                    pass
            t0 = time.time()
            while time.time() - t0 < timeout and self.running():
                time.sleep(0.5)
            if self.running():
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except Exception:
                    pass
            self.proc = None
            self.pid = None
            try:
                self.pidfile.unlink()
            except Exception:
                pass

    def restart(self):
        self.stop()
        time.sleep(1.5)
        return self.start()

    def _watch(self):
        while True:
            time.sleep(5)
            try:
                if self.desired and not self.running():
                    recent = [t for t in self.restarts if time.time() - t < 600]
                    if len(recent) >= 5:
                        self.last_error = "ComfyUI crashed 5 times in 10 minutes; automatic restart paused (see logs/comfy.log)"
                        time.sleep(60)
                        continue
                    self.last_error = "ComfyUI exited unexpectedly; restarting"
                    time.sleep(min(30, 3 * len(recent)))
                    self.start()
            except Exception as e:
                self.last_error = str(e)

    def status(self):
        return {"desired": self.desired, "running": self.running(), "pid": self.pid, "error": self.last_error,
                "restarts_10m": len([t for t in self.restarts if time.time() - t < 600])}


# ------------------------------------------------------------------ async client
class ComfyClient:
    def __init__(self, port=None):
        cfg = load_config()
        self.base = f"http://127.0.0.1:{port or cfg['comfy_port']}"
        self.cid = str(uuid.uuid4())
        self._oi = None
        self._oi_t = 0
        self._session = None

    async def session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, connect=10))
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get_json(self, path, timeout=30):
        s = await self.session()
        async with s.get(self.base + path, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            r.raise_for_status()
            return await r.json()

    async def ready(self):
        try:
            await self._get_json("/system_stats", 5)
            return True
        except Exception:
            return False

    async def system_stats(self):
        return await self._get_json("/system_stats", 10)

    async def object_info(self, node=None, refresh=False):
        if node:
            return await self._get_json(f"/object_info/{node}", 60)
        if self._oi is None or refresh or time.time() - self._oi_t > 120:
            self._oi = await self._get_json("/object_info", 120)
            self._oi_t = time.time()
        return self._oi

    async def supports_node(self, node_name: str) -> bool:
        try:
            return node_name in (await self.object_info())
        except Exception:
            return False

    async def queue(self):
        return await self._get_json("/queue", 10)

    async def upload(self, path):
        s = await self.session()
        data = aiohttp.FormData()
        data.add_field("image", open(path, "rb"), filename=os.path.basename(path), content_type="application/octet-stream")
        data.add_field("overwrite", "true")
        async with s.post(self.base + "/upload/image", data=data, timeout=aiohttp.ClientTimeout(total=1800)) as r:
            r.raise_for_status()
            return (await r.json())["name"]

    async def submit(self, prompt, extra=None):
        s = await self.session()
        body = {"prompt": prompt, "client_id": self.cid}
        if extra:
            body["extra_data"] = extra
        async with s.post(self.base + "/prompt", json=body, timeout=aiohttp.ClientTimeout(total=120)) as r:
            txt = await r.text()
            if r.status != 200:
                try:
                    j = json.loads(txt)
                except Exception:
                    j = {}
                errs = []
                if isinstance(j.get("error"), dict):
                    errs.append(j["error"].get("message", ""))
                for nid, ne in (j.get("node_errors") or {}).items():
                    for e in ne.get("errors", []):
                        errs.append(f"node {nid} ({ne.get('class_type')}): {e.get('message')} {e.get('details', '')}")
                raise PromptRejected("; ".join(x for x in errs if x)[:900] or txt[:500], j)
            return json.loads(txt)["prompt_id"]

    async def history(self, pid):
        return await self._get_json(f"/history/{pid}", 30)

    async def interrupt(self):
        s = await self.session()
        try:
            async with s.post(self.base + "/interrupt", json={}, timeout=aiohttp.ClientTimeout(total=10)):
                pass
        except Exception:
            pass

    async def free(self, unload=False, free_memory=True):
        s = await self.session()
        try:
            async with s.post(self.base + "/free", json={"unload_models": unload, "free_memory": free_memory}, timeout=aiohttp.ClientTimeout(total=30)):
                return True
        except Exception:
            return False

    async def download(self, item, dst):
        from urllib.parse import urlencode
        q = urlencode({"filename": item["filename"], "subfolder": item.get("subfolder", ""), "type": item.get("type", "output")})
        s = await self.session()
        async with s.get(f"{self.base}/view?{q}", timeout=aiohttp.ClientTimeout(total=3600)) as r:
            r.raise_for_status()
            with open(dst, "wb") as f:
                async for chunk in r.content.iter_chunked(1 << 20):
                    f.write(chunk)
        return os.path.getsize(dst)

    async def save_options(self, crf=12):
        """SaveVideo inputs that set a crf on this ComfyUI build, discovered from its published schema."""
        try:
            spec = (await self.object_info("SaveVideo"))["SaveVideo"]["input"]
        except Exception:
            return {}

        def walk(inputs, prefix):
            for name, s in {**inputs.get("required", {}), **inputs.get("optional", {})}.items():
                key = f"{prefix}{name}"
                if name == "crf":
                    return {key: crf}
                if isinstance(s, list) and s and s[0] == "COMFY_DYNAMICCOMBO_V3":
                    for opt in s[1].get("options", []):
                        found = walk(opt.get("inputs", {}), f"{key}.")
                        if found:
                            return {key: opt["key"], **found}
            return {}
        return walk(spec, "")

    async def watch(self, prompt_id, on_event, stop: asyncio.Event):
        """Forward progress websocket events for one prompt to on_event(type, data)."""
        s = await self.session()
        try:
            async with s.ws_connect(f"{self.base}/ws?clientId={self.cid}", heartbeat=20) as ws:
                while not stop.is_set():
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            d = json.loads(msg.data)
                        except Exception:
                            continue
                        data = d.get("data", {}) or {}
                        if data.get("prompt_id") and data["prompt_id"] != prompt_id:
                            continue
                        on_event(d.get("type"), data)
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        return
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
