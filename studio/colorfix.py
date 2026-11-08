"""Colour consistency: flicker detection, colour matching and chunk-seam smoothing.

Everything here is real signal processing on the decoded frames (ffmpeg rawvideo pipes + OpenCV/NumPy):

* analyse()   - per-frame Lab statistics of source and output, flicker metrics, drift and chunk-seam jumps.
* plan()      - turns the statistics into per-frame Lab offsets (source-match / temporal / seam modes).
* apply()     - streams full-resolution frames through the correction (face + skin protection masks) and
                re-encodes ONCE.  Frames whose correction is ~0 are passed through untouched.

Honest limits: the correction is global per frame (mean / optional contrast) weighted by a protection map;
it removes brightness/colour pumping and chunk colour jumps, it does not repair local texture flicker.
Face protection uses OpenCV's frontal Haar cascade, so profile / tiny faces are not detected (skin-tone
protection still covers them partly).
"""
import json, math, os, subprocess, threading
from pathlib import Path

import cv2
import numpy as np

# ------------------------------------------------------------------ ffmpeg raw readers
def _read_exact(f, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            return None if not buf else bytes(buf)
        buf += chunk
    return bytes(buf)


def iter_frames(cmd, w, h):
    """cmd must output rgb24 rawvideo of size w x h to stdout. Yields HxWx3 uint8 arrays."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, bufsize=1 << 24)
    n = w * h * 3
    try:
        while True:
            b = _read_exact(p.stdout, n)
            if b is None or len(b) < n:
                break
            yield np.frombuffer(b, np.uint8).reshape(h, w, 3)
    finally:
        try:
            p.stdout.close()
        except Exception:
            pass
        p.terminate() if p.poll() is None else None
        p.wait()


def lowres_cmd(input_args, vf, w, h, extra_in=None):
    return ["ffmpeg", "-v", "error", *input_args, "-an",
            "-vf", f"{vf + ',' if vf else ''}scale={w}:{h}:flags=area:in_color_matrix=bt709:out_color_matrix=bt709:in_range=tv:out_range=full,format=rgb24",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]


def lab_stats(rgb):
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    m = lab.reshape(-1, 3).mean(0)
    s = lab.reshape(-1, 3).std(0)
    return np.concatenate([m, s])  # L a b sL sa sb


def collect_stats(cmd, w, h, progress=None, total=None):
    out = []
    for i, fr in enumerate(iter_frames(cmd, w, h)):
        out.append(lab_stats(fr))
        if progress and total and i % 8 == 0:
            progress(min(1.0, i / total))
    return np.array(out, np.float32).reshape(-1, 6)


# ------------------------------------------------------------------ metrics
def segments_from_cuts(n, cuts):
    cuts = sorted(set(int(c) for c in cuts if 0 < c < n))
    b = [0] + cuts + [n]
    return list(zip(b[:-1], b[1:]))


def flicker_metric(x, cuts=()):
    """Median |x[t] - mean(x[t-1], x[t+1])| inside scenes: temporal 'pumping' of a global statistic."""
    x = np.asarray(x, np.float64)
    res = []
    for a, b in segments_from_cuts(len(x), cuts):
        if b - a >= 3:
            seg = x[a:b]
            res.append(np.abs(seg[1:-1] - (seg[:-2] + seg[2:]) / 2))
    if not res:
        return 0.0
    r = np.concatenate(res)
    return float(np.median(r))


def seam_jumps(x, bounds, cuts=(), k=3):
    """Jump of a statistic across each chunk boundary, relative to the local frame-to-frame jitter."""
    x = np.asarray(x, np.float64)
    out = []
    cutset = set(int(c) for c in cuts)
    for b in bounds:
        if b < k or b > len(x) - k or any(abs(b - c) <= 1 for c in cutset):
            continue
        a_mean = x[b - k:b].mean()
        b_mean = x[b:b + k].mean()
        lo, hi = max(0, b - 12), min(len(x), b + 12)
        d = np.abs(np.diff(x[lo:hi]))
        jitter = float(np.median(d)) if len(d) else 0.0
        out.append({"frame": int(b), "jump": float(b_mean - a_mean), "jitter": jitter})
    return out


def analyze(src_stats, out_stats, fps, cuts=(), bounds=(), max_points=300):
    n = min(len(src_stats), len(out_stats))
    s, o = src_stats[:n], out_stats[:n]
    fl_src = [flicker_metric(s[:, c], cuts) for c in range(3)]
    fl_out = [flicker_metric(o[:, c], cuts) for c in range(3)]
    f_src_L, f_out_L = fl_src[0], fl_out[0]
    f_src_c, f_out_c = (fl_src[1] + fl_src[2]) / 2, (fl_out[1] + fl_out[2]) / 2
    ratio_L = f_out_L / max(f_src_L, 0.08)
    ratio_c = f_out_c / max(f_src_c, 0.04)
    d = o[:, :3] - s[:, :3]
    drift = {"mean_abs_L": float(np.abs(d[:, 0]).mean()), "mean_abs_a": float(np.abs(d[:, 1]).mean()), "mean_abs_b": float(np.abs(d[:, 2]).mean()),
             "max_abs_L": float(np.abs(d[:, 0]).max()), "max_chroma": float(np.sqrt(d[:, 1] ** 2 + d[:, 2] ** 2).max()),
             "mean_chroma": float(np.sqrt(d[:, 1] ** 2 + d[:, 2] ** 2).mean())}
    sj_L = seam_jumps(o[:, 0], bounds, cuts)
    sj_a = {j["frame"]: j for j in seam_jumps(o[:, 1], bounds, cuts)}
    sj_b = {j["frame"]: j for j in seam_jumps(o[:, 2], bounds, cuts)}
    sj_src_L = {j["frame"]: j for j in seam_jumps(s[:, 0], bounds, cuts)}
    sj_src_a = {j["frame"]: j for j in seam_jumps(s[:, 1], bounds, cuts)}
    sj_src_b = {j["frame"]: j for j in seam_jumps(s[:, 2], bounds, cuts)}
    seams = []
    for j in sj_L:
        fr = j["frame"]
        src_jump_L = sj_src_L.get(fr, {}).get("jump", 0.0)
        excess_L = abs(j["jump"]) - abs(src_jump_L)
        ja = sj_a.get(fr, {"jump": 0.0, "jitter": 0.0})
        jb = sj_b.get(fr, {"jump": 0.0, "jitter": 0.0})
        excess_a = abs(ja["jump"]) - abs(sj_src_a.get(fr, {}).get("jump", 0.0))
        excess_b = abs(jb["jump"]) - abs(sj_src_b.get(fr, {}).get("jump", 0.0))
        excess_c = math.hypot(max(0.0, excess_a), max(0.0, excess_b))
        jitter_c = math.hypot(ja["jitter"], jb["jitter"])
        excess = max(excess_L, excess_c)
        flag_L = excess_L > max(1.0, 2.5 * j["jitter"])
        flag_c = excess_c > max(0.8, 2.5 * jitter_c)
        seams.append({**j, "src_jump": src_jump_L, "excess": float(excess), "excess_chroma": float(excess_c), "flag": bool(flag_L or flag_c)})
    strong = (ratio_L > 1.8 and f_out_L > 0.35) or (ratio_c > 2.2 and f_out_c > 0.25) or any(x["flag"] and x["excess"] > 3 for x in seams)
    mild = (ratio_L > 1.3 and f_out_L > 0.15) or (ratio_c > 1.5 and f_out_c > 0.1) or any(x["flag"] for x in seams) or drift["mean_chroma"] > 1.5 or drift["mean_abs_L"] > 1.5
    verdict = "strong" if strong else ("mild" if mild else "ok")
    step = max(1, n // max_points)
    idx = np.arange(0, n, step)
    series = {"t": (idx / fps).round(3).tolist(), "src_L": s[idx, 0].round(2).tolist(), "out_L": o[idx, 0].round(2).tolist(),
              "chroma_shift": np.sqrt(d[idx, 1] ** 2 + d[idx, 2] ** 2).round(2).tolist()}
    advice = {"ok": "No visible colour pumping or chunk colour jumps detected; the colour pass is not needed.",
              "mild": "Mild colour/brightness pumping or seam steps detected; a colour pass (source match, strength ~0.6) is recommended.",
              "strong": "Strong colour flicker or chunk colour jumps; enable the colour pass (source match) - it will also protect faces."}[verdict]
    return {"frames": int(n), "fps": fps, "verdict": verdict, "advice": advice,
            "flicker": {"src_L": f_src_L, "out_L": f_out_L, "ratio_L": ratio_L, "src_chroma": f_src_c, "out_chroma": f_out_c, "ratio_chroma": ratio_c},
            "drift": drift, "seams": seams, "series": series}


# ------------------------------------------------------------------ correction planning
def _smooth(x, w):
    if w <= 1:
        return x.copy()
    k = np.ones(w) / w
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, k, mode="valid")[:len(x)]


def _median(x, w):
    if w <= 1 or len(x) < 3:
        return x.copy()
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.array([np.median(xp[i:i + w]) for i in range(len(x))])


def plan(src_stats, out_stats, cuts, bounds, mode="source", strength=0.7, window=9, max_shift=6.0, match_std=False):
    """Return dict(offset=(T,3), gain=(T,3)) in Lab units for the output frames."""
    n = min(len(src_stats), len(out_stats))
    s, o = src_stats[:n].astype(np.float64), out_stats[:n].astype(np.float64)
    off = np.zeros((n, 3))
    gain = np.ones((n, 3))
    if mode == "source":
        off = np.clip(s[:, :3] - o[:, :3], -max_shift, max_shift) * strength
        # Also blend out any remaining seam step across chunk boundaries
        o_corr = o[:, :3] + off
        for b in bounds:
            if b < 3 or b > n - 3 or any(abs(b - c) <= 1 for c in cuts):
                continue
            for c in range(3):
                jo = o_corr[b:b + 3, c].mean() - o_corr[b - 3:b, c].mean()
                js = s[b:b + 3, c].mean() - s[b - 3:b, c].mean()
                step = float(np.clip(jo - js, -max_shift, max_shift))
                w = max(2, window)
                for i in range(1, w + 1):
                    t_prev = b - i
                    if t_prev >= 0:
                        off[t_prev, c] += step / 2 * (1 - (i - 1) / w)
                    t_next = b + i - 1
                    if t_next < n:
                        off[t_next, c] -= step / 2 * (1 - (i - 1) / w)
    elif mode == "temporal":
        for a, b in segments_from_cuts(n, cuts):
            if b - a < 3:
                continue
            for c in range(3):
                seg = o[a:b, c]
                target = _smooth(_median(seg, min(window, max(1, b - a) // 2 * 2 + 1)), min(window, max(1, b - a) // 2 * 2 + 1))
                off[a:b, c] = np.clip(target - seg, -max_shift, max_shift) * strength
    elif mode == "seams":
        sj = {j["frame"]: j for j in seam_jumps(s[:, 0], bounds, cuts)}
        for b in bounds:
            if b < 3 or b > n - 3 or any(abs(b - c) <= 1 for c in cuts):
                continue
            for c in range(3):
                jo = o[b:b + 3, c].mean() - o[b - 3:b, c].mean()
                js = s[b:b + 3, c].mean() - s[b - 3:b, c].mean()
                step = (jo - js) * strength
                step = float(np.clip(step, -max_shift, max_shift))
                w = max(2, window)
                for i in range(1, w + 1):
                    t_prev = b - i
                    if t_prev >= 0:
                        off[t_prev, c] += step / 2 * (1 - (i - 1) / w)
                    t_next = b + i - 1
                    if t_next < n:
                        off[t_next, c] -= step / 2 * (1 - (i - 1) / w)
    if match_std and mode == "source":
        for c in range(3):
            g = (s[:, 3 + c] + 1e-3) / (o[:, 3 + c] + 1e-3)
            gain[:, c] = 1 + strength * (np.clip(g, 0.9, 1.1) - 1)
    return {"offset": off, "gain": gain, "mean_abs_offset": float(np.abs(off).mean()), "max_abs_offset": float(np.abs(off).max())}


# ------------------------------------------------------------------ protection masks
_FACE = None


def _face_cascade():
    global _FACE
    if _FACE is None:
        try:
            if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
                p = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
                if os.path.exists(p):
                    c = cv2.CascadeClassifier(p)
                    if not c.empty():
                        _FACE = c
                        return _FACE
        except Exception:
            pass
        _FACE = False
    return _FACE


def protect_map(rgb, faces_hold, protect_faces, protect_skin, frame_idx, det_every=3):
    """Weight map in [0,1] at low resolution: 1 = apply full correction, 0 = leave untouched."""
    h, w = rgb.shape[:2]
    sw = 480 if w > 480 else w
    sh = int(round(h * sw / w))
    small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    prot = np.zeros((sh, sw), np.float32)
    if protect_skin > 0:
        ycc = cv2.cvtColor(small, cv2.COLOR_RGB2YCrCb)
        skin = cv2.inRange(ycc, (0, 133, 77), (255, 173, 127)).astype(np.float32) / 255.0
        skin = cv2.GaussianBlur(skin, (0, 0), 3)
        prot = np.maximum(prot, skin * protect_skin)
    fc = _face_cascade() if protect_faces else None
    if fc:
        if frame_idx % det_every == 0 or "boxes" not in faces_hold:
            gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
            boxes = fc.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(max(24, sw // 40), max(24, sw // 40)))
            faces_hold["boxes"] = [tuple(b) for b in boxes]
        fm = np.zeros((sh, sw), np.float32)
        for (x, y, bw, bh) in faces_hold["boxes"]:
            cx, cy = x + bw / 2, y + bh / 2
            cv2.ellipse(fm, (int(cx), int(cy)), (int(bw * 0.72), int(bh * 0.85)), 0, 0, 360, 1.0, -1)
        if fm.any():
            fm = cv2.GaussianBlur(fm, (0, 0), max(2, sw / 120))
            prot = np.maximum(prot, fm)
    return 1.0 - np.clip(prot, 0, 1), len(faces_hold.get("boxes", []))


def correct_frame(rgb, off, gain, wmap_small):
    """Apply Lab offset/gain to a full-resolution RGB frame, scaled by the protection map."""
    h, w = rgb.shape[:2]
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    wm = cv2.resize(wmap_small, (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
    mean = lab.reshape(-1, 3).mean(0)
    g = np.array(gain, np.float32)
    o = np.array(off, np.float32)
    corrected = (lab - mean) * g + mean + o
    lab = lab + (corrected - lab) * wm
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def apply(decode_cmd, encode_cmd, w, h, plan_, protect_faces=True, protect_skin=0.6, progress=None, total=None, stats=None):
    """Stream decode -> correct -> encode.  Returns number of frames written and how many were corrected."""
    off, gain = plan_["offset"], plan_["gain"]
    enc_log = open(os.devnull, "wb")
    enc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, bufsize=1 << 24)
    err_buf = []
    threading.Thread(target=lambda: err_buf.append(enc.stderr.read()), daemon=True).start()
    hold = {}
    n_corr = n = faces_seen = 0
    try:
        for i, fr in enumerate(iter_frames(decode_cmd, w, h)):
            if i < len(off) and (np.abs(off[i]).max() > 0.05 or np.abs(gain[i] - 1).max() > 0.002):
                wm, nf = protect_map(fr, hold, protect_faces, protect_skin, i)
                faces_seen += 1 if nf else 0
                fr = correct_frame(fr, off[i], gain[i], wm)
                n_corr += 1
            enc.stdin.write(np.ascontiguousarray(fr).tobytes())
            n += 1
            if progress and total and i % 4 == 0:
                progress(min(1.0, i / total))
        enc.stdin.close()
        rc = enc.wait()
    except BrokenPipeError:
        rc = enc.wait()
    if rc != 0:
        raise RuntimeError("colour pass encoder failed: " + (err_buf[0].decode(errors="replace")[-400:] if err_buf else ""))
    return {"frames": n, "corrected": n_corr, "frames_with_faces": faces_seen}
