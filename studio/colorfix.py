"""Colour consistency: overlap crossfade stitching, Wavelet Color Fix, flicker detection,
colour matching and chunk-seam smoothing.

Everything here is real signal processing on the decoded frames (ffmpeg rawvideo pipes + OpenCV/NumPy):

* stitch_chunks_overlap() - streaming cosine-crossfade + overlap-anchored Lab alignment across
                            non-cut chunk boundaries (exact frame count, low RAM).
* analyse()               - per-frame Lab statistics of source and output, flicker metrics, drift
                            and chunk-seam jumps.
* plan()                  - turns statistics into per-frame Lab offsets (wavelet / source / temporal
                            / seams modes), separating chunk-seam equalization from global drift.
* apply_wavelet()         - frequency-separated Wavelet Color Fix in CIE Lab: preserves 100% of
                            LTX-2.5 high-frequency detail/grain while matching local low-frequency
                            colour & lighting to the aligned source video.
* apply()                 - global Lab offset/gain pass for source / temporal / seams modes.
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
    advice = {"ok": "No visible colour pumping or chunk colour jumps detected.",
              "mild": "Mild colour/brightness pumping or seam steps detected; Wavelet Color Fix (strength ~0.85) is recommended.",
              "strong": "Strong colour flicker or chunk colour jumps; enable Wavelet Color Fix to lock local colour & lighting to the source while keeping full detail."}[verdict]
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


def _seam_ramp(o_ref, s_ref, bounds, cuts, window, max_shift, scale=1.0):
    """Compute per-frame Lab offset ramp that removes chunk-boundary steps without affecting cuts."""
    n = min(len(o_ref), len(s_ref))
    seam_off = np.zeros((n, 3), dtype=np.float64)
    if scale <= 0.0:
        return seam_off
    for b in bounds:
        if b < 3 or b > n - 3 or any(abs(b - c) <= 1 for c in cuts):
            continue
        for c in range(3):
            jo = o_ref[b:b + 3, c].mean() - o_ref[b - 3:b, c].mean()
            js = s_ref[b:b + 3, c].mean() - s_ref[b - 3:b, c].mean()
            step = float(np.clip((jo - js) * scale, -max_shift, max_shift))
            w = max(2, window)
            for i in range(1, w + 1):
                t_prev = b - i
                if t_prev >= 0:
                    seam_off[t_prev, c] += step / 2 * (1 - (i - 1) / w)
                t_next = b + i - 1
                if t_next < n:
                    seam_off[t_next, c] -= step / 2 * (1 - (i - 1) / w)
    return seam_off


def plan(src_stats, out_stats, cuts, bounds, mode="wavelet", strength=0.85, window=9, max_shift=12.0, match_std=False):
    """Return dict(offset, global_offset, seam_offset, gain) in Lab units for the output frames."""
    n = min(len(src_stats), len(out_stats))
    s, o = src_stats[:n].astype(np.float64), out_stats[:n].astype(np.float64)
    global_off = np.zeros((n, 3), dtype=np.float64)
    seam_off = np.zeros((n, 3), dtype=np.float64)
    gain = np.ones((n, 3), dtype=np.float64)
    if mode == "wavelet":
        # Wavelet handles local source matching directly; add a residual seam ramp for any
        # uncorrected (1 - strength) step across non-cut chunk boundaries.
        residual = max(0.0, 1.0 - float(strength))
        if residual > 0.01:
            seam_off = _seam_ramp(o[:, :3], s[:, :3], bounds, cuts, window, max_shift, scale=residual)
    elif mode == "source":
        global_off = np.clip(s[:, :3] - o[:, :3], -max_shift, max_shift) * strength
        o_corr = o[:, :3] + global_off
        seam_off = _seam_ramp(o_corr, s[:, :3], bounds, cuts, window, max_shift, scale=1.0)
    elif mode == "temporal":
        for a, b in segments_from_cuts(n, cuts):
            if b - a < 3:
                continue
            for c in range(3):
                seg = o[a:b, c]
                win = min(window, max(1, b - a) // 2 * 2 + 1)
                target = _smooth(_median(seg, win), win)
                global_off[a:b, c] = np.clip(target - seg, -max_shift, max_shift) * strength
    elif mode == "seams":
        seam_off = _seam_ramp(o[:, :3], s[:, :3], bounds, cuts, window, max_shift, scale=strength)
    if match_std and mode == "source":
        for c in range(3):
            g = (s[:, 3 + c] + 1e-3) / (o[:, 3 + c] + 1e-3)
            gain[:, c] = 1 + strength * (np.clip(g, 0.9, 1.1) - 1)
    total_off = global_off + seam_off
    return {
        "offset": total_off,
        "global_offset": global_off,
        "seam_offset": seam_off,
        "gain": gain,
        "mean_abs_offset": float(np.abs(total_off).mean()),
        "max_abs_offset": float(np.abs(total_off).max()),
    }


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


def correct_frame(rgb, off, gain, wmap_small, seam_off=None):
    """Apply Lab offset/gain to a full-resolution RGB frame.

    ``off`` and ``gain`` respect ``wmap_small`` (face/skin protection), while ``seam_off`` is
    applied uniformly across the frame so faces never pop relative to the scene at chunk cuts.
    """
    h, w = rgb.shape[:2]
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    wm = cv2.resize(wmap_small, (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
    mean = lab.reshape(-1, 3).mean(0)
    g = np.array(gain, np.float32)
    o = np.array(off, np.float32)
    corrected = (lab - mean) * g + mean + o
    lab = lab + (corrected - lab) * wm
    if seam_off is not None:
        so = np.array(seam_off, np.float32)
        if np.abs(so).max() > 1e-4:
            lab = lab + so
    lab[..., 0] = np.clip(lab[..., 0], 0.0, 100.0)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _apply_uniform_lab_shift(rgb, delta_lab):
    """Fast uniform Lab offset applied to a uint8 RGB frame (used for overlap seam alignment)."""
    d = np.asarray(delta_lab, dtype=np.float32)
    if np.abs(d).max() <= 0.05:
        return rgb
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    lab += d
    lab[..., 0] = np.clip(lab[..., 0], 0.0, 100.0)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _mean_lab_frames(frames, sample_w=320):
    """Average CIE Lab mean across a short list of uint8 RGB frames."""
    acc = np.zeros(3, dtype=np.float64)
    for fr in frames:
        h, w = fr.shape[:2]
        if w > sample_w:
            sh = max(2, int(round(h * sample_w / w)))
            small = cv2.resize(fr, (sample_w, sh), interpolation=cv2.INTER_AREA)
        else:
            small = fr
        lab = cv2.cvtColor(small.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        acc += lab.reshape(-1, 3).mean(0)
    return acc / max(1, len(frames))


def cosine_blend_frames(prev_tail, curr_head, align_color=True, max_align_shift=8.0):
    """Cosine-crossfade two equal-length overlapping frame lists, optionally aligning chunk color first.

    Returns ``(blended_frames, delta_lab)`` where ``delta_lab`` is the 3-element Lab offset that
    aligned ``curr_head`` onto ``prev_tail`` (so callers can decay it smoothly into the next chunk).
    """
    ov = min(len(prev_tail), len(curr_head))
    if ov <= 0:
        return [], np.zeros(3, dtype=np.float64)
    delta_lab = np.zeros(3, dtype=np.float64)
    if align_color:
        raw_delta = _mean_lab_frames(prev_tail[:ov]) - _mean_lab_frames(curr_head[:ov])
        delta_lab = np.clip(raw_delta, -float(max_align_shift), float(max_align_shift))
    blended = []
    for t in range(ov):
        alpha = 0.5 * (1.0 - math.cos(math.pi * (t + 1) / (ov + 1)))
        a_fr = prev_tail[t].astype(np.float32)
        b_u8 = _apply_uniform_lab_shift(curr_head[t], delta_lab) if align_color else curr_head[t]
        b_fr = b_u8.astype(np.float32)
        mix = a_fr * (1.0 - alpha) + b_fr * alpha
        blended.append(np.clip(mix + 0.5, 0, 255).astype(np.uint8))
    return blended, delta_lab


def stitch_chunks_overlap(chunk_specs, encode_cmd, w, h, align_color=True, max_align_shift=8.0, decay_frames=24, progress=None, total=None):
    """Stream-stitch multiple chunk videos with cosine crossfade across overlapping boundaries.

    Each item in ``chunk_specs`` must have ``path``, ``extract_frames``, and ``overlap_prev``.
    Emits exact ``sum(extract_frames - overlap_prev)`` frames into ``encode_cmd`` stdin while
    keeping only ``overlap_prev`` frames buffered in RAM at any time.
    """
    enc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, bufsize=1 << 24)
    err_buf = []
    t_err = threading.Thread(target=lambda: err_buf.append(enc.stderr.read()), daemon=True)
    t_err.start()

    emitted = 0
    prev_tail = []
    seams_blended = 0
    try:
        for idx, spec in enumerate(chunk_specs):
            c_path = str(spec["path"])
            extract_f = int(spec["extract_frames"])
            ov_prev = int(spec.get("overlap_prev", 0)) if idx > 0 else 0
            ov_next = int(chunk_specs[idx + 1].get("overlap_prev", 0)) if idx + 1 < len(chunk_specs) else 0
            if ov_prev + ov_next > extract_f:
                raise ValueError(f"Chunk {idx}: overlap_prev ({ov_prev}) + overlap_next ({ov_next}) > extract_frames ({extract_f})")

            dec_cmd = [
                "ffmpeg", "-v", "error", "-i", c_path, "-an",
                "-vf", "scale=in_color_matrix=bt709:out_color_matrix=bt709:in_range=tv:out_range=full,format=rgb24",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
            ]

            curr_head = []
            next_tail = []
            delta_lab = np.zeros(3, dtype=np.float64)
            interior_len = extract_f - ov_prev
            decay_len = min(int(decay_frames), max(1, interior_len))
            seen = 0

            for fr in iter_frames(dec_cmd, w, h):
                if seen >= extract_f:
                    break
                if seen < ov_prev:
                    curr_head.append(fr.copy())
                    if len(curr_head) == ov_prev:
                        blended, delta_lab = cosine_blend_frames(
                            prev_tail, curr_head, align_color=align_color, max_align_shift=max_align_shift
                        )
                        for bfr in blended:
                            enc.stdin.write(np.ascontiguousarray(bfr).tobytes())
                            emitted += 1
                        prev_tail = []
                        curr_head = []
                        seams_blended += 1
                else:
                    t_after = seen - ov_prev
                    out_fr = fr
                    if align_color and np.abs(delta_lab).max() > 0.05 and t_after < decay_len:
                        w_decay = 0.5 * (1.0 + math.cos(math.pi * (t_after + 1) / (decay_len + 1)))
                        out_fr = _apply_uniform_lab_shift(fr, delta_lab * w_decay)
                    if ov_next > 0 and seen >= extract_f - ov_next:
                        next_tail.append(out_fr.copy())
                    else:
                        enc.stdin.write(np.ascontiguousarray(out_fr).tobytes())
                        emitted += 1
                seen += 1
                if progress and total and emitted % 8 == 0:
                    progress(min(1.0, emitted / total))

            if seen < extract_f:
                raise RuntimeError(f"Chunk {idx} decoded {seen} frames, expected {extract_f}")
            prev_tail = next_tail

        enc.stdin.close()
        rc = enc.wait()
    except BrokenPipeError:
        rc = enc.wait()
    finally:
        try:
            if enc.stdin and not enc.stdin.closed:
                enc.stdin.close()
        except Exception:
            pass
        if enc.poll() is None:
            enc.terminate()
            enc.wait()
        t_err.join(timeout=5)
        try:
            if enc.stderr and not enc.stderr.closed:
                enc.stderr.close()
        except Exception:
            pass

    if rc != 0:
        raise RuntimeError("overlap stitch encoder failed: " + (err_buf[0].decode(errors="replace")[-400:] if err_buf else ""))
    return {"frames": emitted, "seams_blended": seams_blended}


def wavelet_color_fix_frame(out_rgb, src_small_rgb, strength=0.85, max_shift=12.0, radius_frac=0.035, seam_off=None, wmap_small=None):
    """Frequency-separated Wavelet Color Fix in CIE Lab.

    Replaces only the smooth low-frequency Lab colour & illumination field of ``out_rgb`` toward
    ``src_small_rgb`` (the aligned source frame) while leaving 100% of high-frequency LTX-2.5
    micro-texture, pores, edges and grain untouched.
    """
    strength = float(strength)
    has_seam = seam_off is not None and float(np.abs(seam_off).max()) > 1e-4
    if strength <= 1e-4 and not has_seam:
        return out_rgb

    h, w = out_rgb.shape[:2]
    sh, sw = src_small_rgb.shape[:2]

    out_small = out_rgb if (h == sh and w == sw) else cv2.resize(out_rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    out_lab_s = cv2.cvtColor(out_small.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    src_lab_s = cv2.cvtColor(src_small_rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)

    diff_s = src_lab_s - out_lab_s
    sigma = max(3.0, float(sw) * float(radius_frac))
    diff_low = cv2.GaussianBlur(diff_s, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    diff_low = np.clip(diff_low, -float(max_shift), float(max_shift)) * strength

    if wmap_small is not None:
        wm_s = wmap_small if wmap_small.shape[:2] == (sh, sw) else cv2.resize(wmap_small, (sw, sh), interpolation=cv2.INTER_LINEAR)
        diff_low = diff_low * wm_s[..., None]

    if has_seam:
        diff_low = diff_low + np.asarray(seam_off, dtype=np.float32)

    diff_full = diff_low if (h == sh and w == sw) else cv2.resize(diff_low, (w, h), interpolation=cv2.INTER_LINEAR)

    lab = cv2.cvtColor(out_rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    lab += diff_full
    lab[..., 0] = np.clip(lab[..., 0], 0.0, 100.0)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def apply_wavelet(decode_out_cmd, decode_src_cmd, encode_cmd, w, h, sw, sh, strength=0.85, max_shift=12.0, radius_frac=0.035, seam_offsets=None, protect_faces=False, protect_skin=0.0, progress=None, total=None):
    """Stream full-res enhanced output + aligned low-res source through Wavelet Color Fix and encode ONCE."""
    enc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, bufsize=1 << 24)
    err_buf = []
    t_err = threading.Thread(target=lambda: err_buf.append(enc.stderr.read()), daemon=True)
    t_err.start()

    src_gen = iter_frames(decode_src_cmd, sw, sh)
    last_src = None
    hold = {}
    use_prot = bool(protect_faces) or float(protect_skin) > 0.0
    n = n_corr = faces_seen = 0
    try:
        for i, fr in enumerate(iter_frames(decode_out_cmd, w, h)):
            try:
                src_s = next(src_gen)
                last_src = src_s
            except StopIteration:
                src_s = last_src
            if src_s is not None:
                wm = None
                if use_prot:
                    wm, nf = protect_map(fr, hold, protect_faces, protect_skin, i)
                    faces_seen += 1 if nf else 0
                so = seam_offsets[i] if (seam_offsets is not None and i < len(seam_offsets)) else None
                fr = wavelet_color_fix_frame(
                    fr, src_s, strength=strength, max_shift=max_shift,
                    radius_frac=radius_frac, seam_off=so, wmap_small=wm,
                )
                n_corr += 1
            enc.stdin.write(np.ascontiguousarray(fr).tobytes())
            n += 1
            if progress and total and i % 4 == 0:
                progress(min(1.0, i / total))
        enc.stdin.close()
        rc = enc.wait()
    except BrokenPipeError:
        rc = enc.wait()
    finally:
        try:
            src_gen.close()
        except Exception:
            pass
        try:
            if enc.stdin and not enc.stdin.closed:
                enc.stdin.close()
        except Exception:
            pass
        if enc.poll() is None:
            enc.terminate()
            enc.wait()
        t_err.join(timeout=5)
        try:
            if enc.stderr and not enc.stderr.closed:
                enc.stderr.close()
        except Exception:
            pass

    if rc != 0:
        raise RuntimeError("wavelet colour pass encoder failed: " + (err_buf[0].decode(errors="replace")[-400:] if err_buf else ""))
    return {"frames": n, "corrected": n_corr, "frames_with_faces": faces_seen}


def apply(decode_cmd, encode_cmd, w, h, plan_, protect_faces=False, protect_skin=0.0, progress=None, total=None, stats=None):
    """Stream decode -> correct -> encode.  Returns number of frames written and how many were corrected."""
    off = plan_.get("global_offset", plan_["offset"])
    seam_off = plan_.get("seam_offset")
    gain = plan_["gain"]
    enc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, bufsize=1 << 24)
    err_buf = []
    t_err = threading.Thread(target=lambda: err_buf.append(enc.stderr.read()), daemon=True)
    t_err.start()
    hold = {}
    use_prot = bool(protect_faces) or float(protect_skin) > 0.0
    n_corr = n = faces_seen = 0
    try:
        for i, fr in enumerate(iter_frames(decode_cmd, w, h)):
            has_global = i < len(off) and (np.abs(off[i]).max() > 0.05 or np.abs(gain[i] - 1).max() > 0.002)
            has_seam = seam_off is not None and i < len(seam_off) and np.abs(seam_off[i]).max() > 0.05
            if has_global or has_seam:
                if use_prot and has_global:
                    wm, nf = protect_map(fr, hold, protect_faces, protect_skin, i)
                    faces_seen += 1 if nf else 0
                else:
                    wm = np.ones((2, 2), dtype=np.float32)
                fr = correct_frame(
                    fr,
                    off[i] if i < len(off) else np.zeros(3),
                    gain[i] if i < len(gain) else np.ones(3),
                    wm,
                    seam_off=seam_off[i] if has_seam else None,
                )
                n_corr += 1
            enc.stdin.write(np.ascontiguousarray(fr).tobytes())
            n += 1
            if progress and total and i % 4 == 0:
                progress(min(1.0, i / total))
        enc.stdin.close()
        rc = enc.wait()
    except BrokenPipeError:
        rc = enc.wait()
    finally:
        try:
            if enc.stdin and not enc.stdin.closed:
                enc.stdin.close()
        except Exception:
            pass
        if enc.poll() is None:
            enc.terminate()
            enc.wait()
        t_err.join(timeout=5)
        try:
            if enc.stderr and not enc.stderr.closed:
                enc.stderr.close()
        except Exception:
            pass
    if rc != 0:
        raise RuntimeError("colour pass encoder failed: " + (err_buf[0].decode(errors="replace")[-400:] if err_buf else ""))
    return {"frames": n, "corrected": n_corr, "frames_with_faces": faces_seen}
