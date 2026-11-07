"""Video processing: metadata probing, scene cuts detection, timeline thumbnails,
geometry solvers (respecting ReDetail's 32/64 grids), smart chunking and ffmpeg assembly."""
import json, math, os, re, subprocess
from pathlib import Path

# Grid constraints from ReDetail 2.0:
# Refine guide is full resolution and requires /32.
# Pixel guide is half resolution with 2x2 latent patches and requires /64.
GRID = {"refine": 32, "restore": 32, "pixel": 64}


def run_cmd(cmd, timeout=3600):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout)
    return r.stdout


def probe_video(video_path: str):
    video_path = str(video_path)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Query video stream
    raw = run_cmd([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,codec_name,pix_fmt",
        "-show_entries", "format=duration,size,bit_rate",
        "-of", "json", video_path
    ])
    data = json.loads(raw or "{}")
    streams = data.get("streams", [])
    if not streams:
        raise ValueError(f"No video streams found in {video_path}")
    st = streams[0]
    fmt = data.get("format", {})

    width = int(st.get("width", 1920))
    height = int(st.get("height", 1080))
    codec = st.get("codec_name", "h264")
    pix_fmt = st.get("pix_fmt", "yuv420p")

    # Accurate FPS
    r_fps = st.get("r_frame_rate", "24/1")
    if "/" in r_fps:
        num, den = r_fps.split("/")
        fps = round(float(num) / float(den or 1), 3)
    else:
        fps = float(r_fps or 24.0)

    # Accurate frame count
    raw_nb = str(st.get("nb_frames") or "")
    nb = int(raw_nb) if raw_nb.isdigit() else 0
    if not nb:
        # Fallback to exact frame count
        c = run_cmd([
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", video_path
        ]).strip()
        nb = int(c) if c.isdigit() else 0

    dur_fmt = float(fmt.get("duration", 0.0) or 0.0)
    dur = round(nb / fps, 3) if nb and fps else dur_fmt

    # Query audio stream
    raw_a = run_cmd([
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,channels,sample_rate,bit_rate",
        "-of", "json", video_path
    ])
    data_a = json.loads(raw_a or "{}")
    a_streams = data_a.get("streams", [])
    has_audio = len(a_streams) > 0
    audio_info = {}
    if has_audio:
        ast = a_streams[0]
        audio_info = {
            "codec": ast.get("codec_name"),
            "channels": ast.get("channels"),
            "sample_rate": ast.get("sample_rate"),
            "bit_rate": ast.get("bit_rate")
        }

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "r_frame_rate": r_fps,
        "frames": nb or max(1, round(dur * fps)),
        "duration": dur,
        "codec": codec,
        "pix_fmt": pix_fmt,
        "size_bytes": int(fmt.get("size", 0) or os.path.getsize(video_path)),
        "has_audio": has_audio,
        "audio": audio_info
    }


def detect_scene_cuts(video_path: str, threshold: float = 0.3):
    """Detects scene transition timestamps (seconds) using ffmpeg scene detection filter."""
    video_path = str(video_path)
    r = subprocess.run([
        "ffmpeg", "-v", "info", "-i", video_path, "-an",
        "-vf", f"select='gt(scene,{threshold})',metadata=print",
        "-f", "null", "-"
    ], capture_output=True, text=True)
    timestamps = sorted(float(m) for m in re.findall(r"pts_time:([\d.]+)", r.stderr))
    return timestamps


def generate_timeline_thumbnails(video_path: str, out_dir: str, count: int = 12):
    """Generates evenly spaced lightweight thumbnail sprites for the interactive timeline."""
    os.makedirs(out_dir, exist_ok=True)
    info = probe_video(video_path)
    dur = info["duration"]
    if dur <= 0:
        return []

    count = max(4, min(count, 30))
    step = dur / count
    thumbs = []
    for i in range(count):
        t = round(i * step, 2)
        thumb_name = f"thumb_{i:03d}.jpg"
        thumb_path = os.path.join(out_dir, thumb_name)
        subprocess.run([
            "ffmpeg", "-y", "-v", "error",
            "-ss", str(t), "-i", video_path,
            "-frames:v", "1", "-vf", "scale=160:-1:flags=fast_bilinear",
            "-q:v", "5", thumb_path
        ], check=False)
        if os.path.exists(thumb_path):
            thumbs.append({"time_s": t, "index": i, "filename": thumb_name})
    return thumbs


# ---------------------------------------------------------------- Geometry & Sizing
def fit_source(w, h, max_crop=0.03):
    """Fits source on the /64 grid (pixel mode requirement)."""
    cw, ch = (w // 64) * 64, (h // 64) * 64
    if (cw, ch) == (w, h):
        return "none", w, h
    if max(1 - cw / w, 1 - ch / h) <= max_crop:
        return "crop", cw, ch
    ar, best = w / h, None
    for b in range(1, 65):
        fh = b * 64
        fw = max(64, round(fh * ar / 64) * 64)
        if abs((fw / fh) / ar - 1) > 0.005:
            continue
        if not 0.65 <= fh / h <= 2.0:
            continue
        d = abs(fh / h - 1)
        if best is None or d < best[0]:
            best = (d, fw, fh)
    return ("resize", best[1], best[2]) if best else ("crop", cw, ch)


def solve_dims(w, h, scale, grid=64):
    """Finds nearest on-grid target dimensions that preserve aspect ratio with minimal distortion."""
    ar = w / h
    best = None
    lo = max(grid, int(scale * h * 0.85) // grid * grid)
    hi = int(scale * h * 1.15) // grid * grid + grid + 1
    for th in range(lo, hi, grid):
        tw = max(grid, round(th * ar / grid) * grid)
        aerr = abs((tw / th) / ar - 1)
        if aerr > 0.02:
            continue
        serr = ((tw / w) + (th / h)) / 2 / scale - 1
        score = aerr * 8 + abs(serr) * (1.3 if serr > 0 else 1.0)
        if best is None or score < best[0]:
            best = (score, tw, th, aerr, serr)
    if best is None:
        # Fallback to closest grid multiple
        tw = max(grid, round(w * scale / grid) * grid)
        th = max(grid, round(h * scale / grid) * grid)
        return tw, th, abs((tw / th) / ar - 1), 0.0
    return best[1], best[2], best[3], best[4]


def target_resolution(orig_w, orig_h, mode="refine", resolution_mode="scale", scale=1.5, preset_long_edge=1920):
    """Calculates optimal target width and height according to mode and grid requirements."""
    grid = GRID.get(mode, 32)
    refine = (mode != "pixel")

    if resolution_mode == "preset":
        # Calculate scale factor from requested preset long edge
        cur_long = max(orig_w, orig_h)
        eff_scale = float(preset_long_edge) / float(cur_long)
        scale = max(1.0, eff_scale)

    if refine:
        fit = "crop" if (orig_w % 2 or orig_h % 2) else "none"
        cw = (orig_w // 2) * 2 if fit == "crop" else orig_w
        ch = (orig_h // 2) * 2 if fit == "crop" else orig_h
    else:
        fit, cw, ch = fit_source(orig_w, orig_h)

    eff = (orig_h * scale) / ch
    tw, th, aerr, serr = solve_dims(cw, ch, eff, grid)

    return {
        "fit": fit, "cw": cw, "ch": ch,
        "tw": tw, "th": th,
        "aspect_error": round(aerr * 100, 2),
        "scale_actual": round(((tw / orig_w) + (th / orig_h)) / 2, 2),
        "grid": grid
    }


# ---------------------------------------------------------------- Smart Chunking
def render_length_8n1(n: int) -> int:
    """Computes smallest 8n+1 >= n for LTX temporal latent window alignment."""
    return n if n % 8 == 1 else ((n - 1) // 8 + 1) * 8 + 1


def smart_chunking(total_frames: int, fps: float, scene_cuts_s: list, budget_fmp: float, tw: int, th: int, refine: bool = True, max_chunk_s: float = 0.0, min_sec: float = 2.0):
    """Tiles video into contiguous chunks snapped to scene cuts where possible, all length 8n+1."""
    dur = total_frames / fps
    minf = max(9, int(min_sec * fps))

    # Calculate max duration per chunk from memory budget (frame-megapixels)
    mp_per_frame = (tw * th) / 1e6
    extra_reserve = 7 + (8 if refine else 0)
    maxf_budget = max(9.0, (budget_fmp / mp_per_frame) - extra_reserve)

    if max_chunk_s > 0:
        maxf_user = int(max_chunk_s * fps)
        maxf = min(maxf_budget, maxf_user)
    else:
        maxf = maxf_budget

    maxf = max(minf, int(maxf))

    cutf = [round(c * fps) for c in scene_cuts_s if 0.1 < c < dur - 0.1]

    k_vram = max(1, -(-total_frames // maxf))
    k = k_vram

    bounds = [0]
    for i in range(1, k):
        t = round(i * total_frames / k)
        remaining = k - i
        near = [
            c for c in cutf
            if abs(c - t) <= maxf * 0.35 and c >= bounds[-1] + minf and c <= total_frames - minf
            and c - bounds[-1] <= maxf
            and total_frames - c <= maxf * remaining
        ]
        bounds.append(min(near, key=lambda c: abs(c - t)) if near else t)
    bounds.append(total_frames)

    chunks = []
    for idx, (a, b) in enumerate(zip(bounds, bounds[1:])):
        chunk_frames = b - a
        start_s = a / fps
        end_s = b / fps
        chunks.append({
            "index": idx,
            "start_frame": a,
            "end_frame": b,
            "frames": chunk_frames,
            "start_s": round(start_s, 3),
            "end_s": round(end_s, 3),
            "duration_s": round(end_s - start_s, 3),
            "render_frames": render_length_8n1(chunk_frames) + (8 if refine else 0)
        })

    return chunks
