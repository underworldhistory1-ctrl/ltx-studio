"""Job settings: defaults, validation and the UI schema (the single source of truth).

Every field listed in SCHEMA is consumed by the engine; the UI renders its controls from this
schema, so there is no control without an implementation behind it.
"""
import copy, re

MODES = {
    "refine": "Refine Details - keeps framing, colour, motion and faces; adds fine texture (default)",
    "restore": "Restore - cleans compression damage / noise on old footage (Restore IC-LoRA)",
    "pixel": "Pixel Upscaler - repaints harder, invents more detail, ~2x faster; faces may drift",
}

DISTILLED_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"

DEFAULTS = {
    "mode": "refine", "restore_first": False,
    "resolution_mode": "scale", "scale": 1.5, "preset_long_edge": 1920,
    "guide_strength": 1.0, "lora_strength": 1.0,
    "seed": 42, "randomize_seed": False,
    "precision": "int8", "gguf_file": "LTX-2.5-Distilled-Q4_K_M.gguf",
    "cached_cond": True, "encoder": "", "clip_device": "default",
    "prompt_positive": "", "prompt_negative": "",
    "sampler_name": "euler", "sigmas_mode": "distilled", "sigmas_custom": DISTILLED_SIGMAS,
    "tile_mode": "auto", "tile_w": 1024, "tile_h": 576, "overlap_frac": 0.0, "blend_var": 0.05,
    "grid_cycle": 1, "tile_frames": 0, "canvas_device": "auto",
    "video_vae": "ltx-2.5-video-vae-bf16.safetensors", "audio_vae": "ltx-2.5-audio-vae-bf16.safetensors",
    "decode_tile": 1024, "decode_temporal": 0,
    "budget_fmp": 0, "max_chunk_s": 0, "snap_to_cuts": True,
    "audio_mode": "original", "audio_bitrate_k": 256,
    "container": "mp4", "vcodec": "libx264", "crf": 14, "x_preset": "slow", "pix_fmt": "yuv420p", "faststart": True,
    "keep_chunks": False,
    "color_enabled": True, "color_mode": "source", "color_strength": 0.85, "color_window": 9,
    "color_max_shift": 6.0, "color_match_std": True, "protect_faces": True, "protect_skin": 0.6,
    "color_auto": False,
    "range_start_s": None, "range_end_s": None,
    "test_enabled": False, "test_start_s": 0.0, "test_frames": 17,
}

OPTIONS = {
    "mode": list(MODES),
    "resolution_mode": ["scale", "preset"],
    "preset_long_edge": [1280, 1920, 2560, 3840],
    "precision": ["int8", "gguf"],
    "clip_device": ["default", "cpu"],
    "sigmas_mode": ["distilled", "custom"],
    "tile_mode": ["auto", "manual"],
    "canvas_device": ["auto", "cpu", "gpu"],
    "audio_mode": ["original", "generated", "none"],
    "container": ["mp4", "mov", "mkv"],
    "vcodec": ["libx264", "libx265"],
    "x_preset": ["veryfast", "fast", "medium", "slow", "slower"],
    "pix_fmt": ["yuv420p", "yuv420p10le"],
    "color_mode": ["source", "temporal", "seams"],
    "sampler_name": ["euler", "euler_ancestral", "heun", "dpm_2", "lms", "ddim", "uni_pc"],
    "test_frames": [9, 17, 25, 33, 49, 65],
}

# group, key, label, type, extra, applies_to modes, advanced
SCHEMA = [
    {"group": "engine", "title": "Engine", "fields": [
        {"key": "mode", "label": "Mode", "type": "cards", "options": OPTIONS["mode"], "labels": MODES},
        {"key": "restore_first", "label": "Restore first, then Refine (2 passes)", "type": "bool", "modes": ["refine"],
         "help": "Lightricks recommends running Restore before Refine on archive/broadcast footage."},
        {"key": "guide_strength", "label": "Guide strength", "type": "range", "min": 0.3, "max": 1.0, "step": 0.05, "modes": ["refine", "restore"],
         "help": "How hard the guide holds to your clip. Below ~0.6 faces start to drift (0.4 swapped identity in the ReDetail test)."},
        {"key": "lora_strength", "label": "IC-LoRA strength", "type": "range", "min": 0.0, "max": 1.5, "step": 0.05},
        {"key": "seed", "label": "Seed", "type": "number", "min": 0, "max": 2**31 - 1, "step": 1},
        {"key": "randomize_seed", "label": "New random seed for every job", "type": "bool"},
    ]},
    {"group": "resolution", "title": "Resolution & Upscale", "fields": [
        {"key": "resolution_mode", "label": "Target by", "type": "select", "options": OPTIONS["resolution_mode"]},
        {"key": "scale", "label": "Scale factor", "type": "range", "min": 1.0, "max": 2.0, "step": 0.05, "show_if": {"resolution_mode": "scale"},
         "help": "1.0 = re-detail in place, 1.5 = sweet spot, 2.0 = heaviest. 1080p x2 lands in the 4K class."},
        {"key": "preset_long_edge", "label": "Output long edge (px)", "type": "select", "options": OPTIONS["preset_long_edge"], "show_if": {"resolution_mode": "preset"}},
    ]},
    {"group": "model", "title": "Model & Precision", "fields": [
        {"key": "precision", "label": "Transformer", "type": "select", "options": OPTIONS["precision"],
         "labels": {"int8": "INT8 ConvRot (CUDA kernels, default)", "gguf": "GGUF Q4_K_M (low VRAM)"}},
        {"key": "gguf_file", "label": "GGUF file", "type": "select", "options_from": "ggufs", "show_if": {"precision": "gguf"}},
        {"key": "cached_cond", "label": "Cached conditioning (skip text encoder)", "type": "bool",
         "help": "Bit-identical for the built-in prompts, saves ~15 GB and time. Turned off automatically when you edit the prompts."},
        {"key": "encoder", "label": "Text encoder", "type": "select", "options_from": "encoders", "show_if": {"cached_cond": False}, "advanced": True},
        {"key": "clip_device", "label": "Text encoder device", "type": "select", "options": OPTIONS["clip_device"], "show_if": {"cached_cond": False}, "advanced": True},
    ]},
    {"group": "prompts", "title": "Prompts", "fields": [
        {"key": "prompt_positive", "label": "Positive prompt (empty = ReDetail default)", "type": "textarea", "modes": ["refine", "restore"]},
        {"key": "prompt_negative", "label": "Negative prompt (empty = ReDetail default)", "type": "textarea", "modes": ["refine", "restore"]},
    ]},
    {"group": "sampler", "title": "Sampler", "advanced": True, "fields": [
        {"key": "sampler_name", "label": "Sampler", "type": "select", "options": OPTIONS["sampler_name"], "default": "euler"},
        {"key": "sigmas_mode", "label": "Sigma schedule", "type": "select", "options": OPTIONS["sigmas_mode"],
         "labels": {"distilled": "Distilled 8-step (default)", "custom": "Custom"}},
        {"key": "sigmas_custom", "label": "Custom sigmas (descending, end at 0)", "type": "text", "show_if": {"sigmas_mode": "custom"}},
    ]},
    {"group": "tiles", "title": "Tiles (refine / restore)", "advanced": True, "modes": ["refine", "restore"], "fields": [
        {"key": "tile_mode", "label": "Tile size", "type": "select", "options": OPTIONS["tile_mode"], "labels": {"auto": "Trained 1024x576 (recommended)", "manual": "Manual"}},
        {"key": "tile_w", "label": "Tile width", "type": "number", "min": 256, "max": 2048, "step": 32, "show_if": {"tile_mode": "manual"}},
        {"key": "tile_h", "label": "Tile height", "type": "number", "min": 256, "max": 2048, "step": 32, "show_if": {"tile_mode": "manual"}},
        {"key": "overlap_frac", "label": "Tile overlap (0 = graph default)", "type": "range", "min": 0.0, "max": 0.5, "step": 0.01},
        {"key": "blend_var", "label": "Blend variance", "type": "range", "min": 0.0, "max": 0.5, "step": 0.01},
        {"key": "grid_cycle", "label": "Grid cycle", "type": "number", "min": 1, "max": 8, "step": 1},
        {"key": "tile_frames", "label": "Frames per window (0 = graph default)", "type": "number", "min": 0, "max": 241, "step": 8},
        {"key": "canvas_device", "label": "Canvas device", "type": "select", "options": OPTIONS["canvas_device"]},
    ]},
    {"group": "vae", "title": "VAE & Decode", "advanced": True, "fields": [
        {"key": "video_vae", "label": "Video VAE", "type": "select", "options_from": "vaes"},
        {"key": "audio_vae", "label": "Audio VAE", "type": "select", "options_from": "vaes"},
        {"key": "decode_tile", "label": "Decode tile size (0 or 1024 = full 1024)", "type": "number", "min": 0, "max": 1024, "step": 32,
         "help": "1024 prevents spatial VAE tile seams. Lower to 256 if decoding OOMs on 24 GB cards."},
        {"key": "decode_temporal", "label": "Decode temporal size (0 = default 128)", "type": "number", "min": 0, "max": 256, "step": 8},
    ]},
    {"group": "memory", "title": "Memory Budget & Chunking", "advanced": True, "fields": [
        {"key": "budget_fmp", "label": "Budget, frame-megapixels per chunk (0 = auto from RAM)", "type": "number", "min": 0, "max": 2000, "step": 10,
         "help": "Refine: the system-RAM dial (~33 MB per frame-MP on top of the staged models). Pixel: the VRAM dial."},
        {"key": "max_chunk_s", "label": "Max chunk length, seconds (0 = auto)", "type": "number", "min": 0, "max": 60, "step": 0.5},
        {"key": "snap_to_cuts", "label": "Snap chunk boundaries to scene cuts", "type": "bool"},
    ]},
    {"group": "audio", "title": "Audio", "fields": [
        {"key": "audio_mode", "label": "Audio", "type": "select", "options": OPTIONS["audio_mode"],
         "labels": {"original": "Original track, re-muxed losslessly in sync", "generated": "Model-generated (not recommended)", "none": "No audio"}},
        {"key": "audio_bitrate_k", "label": "Audio bitrate (kbps)", "type": "number", "min": 96, "max": 512, "step": 32, "advanced": True},
    ]},
    {"group": "export", "title": "Export", "fields": [
        {"key": "container", "label": "Container", "type": "select", "options": OPTIONS["container"]},
        {"key": "vcodec", "label": "Codec", "type": "select", "options": OPTIONS["vcodec"]},
        {"key": "crf", "label": "Quality (CRF, lower = better)", "type": "range", "min": 8, "max": 28, "step": 1},
        {"key": "x_preset", "label": "Encoder preset", "type": "select", "options": OPTIONS["x_preset"], "advanced": True},
        {"key": "pix_fmt", "label": "Pixel format", "type": "select", "options": OPTIONS["pix_fmt"], "advanced": True},
        {"key": "faststart", "label": "Fast-start (web playback)", "type": "bool", "advanced": True},
        {"key": "keep_chunks", "label": "Keep per-chunk intermediates", "type": "bool", "advanced": True},
    ]},
    {"group": "color", "title": "Color Consistency", "fields": [
        {"key": "color_enabled", "label": "Enable colour consistency pass", "type": "bool",
         "help": "Runs after rendering: reduces colour/brightness flicker and chunk-to-chunk colour jumps."},
        {"key": "color_auto", "label": "Only apply when flicker is detected", "type": "bool", "show_if": {"color_enabled": True}},
        {"key": "color_mode", "label": "Match mode", "type": "select", "options": OPTIONS["color_mode"], "show_if": {"color_enabled": True},
         "labels": {"source": "Match the source clip (colour matching)", "temporal": "Only remove temporal flicker", "seams": "Only smooth chunk seams"}},
        {"key": "color_strength", "label": "Strength", "type": "range", "min": 0.0, "max": 1.0, "step": 0.05, "show_if": {"color_enabled": True}},
        {"key": "color_window", "label": "Temporal smoothing window (frames)", "type": "number", "min": 1, "max": 61, "step": 2, "show_if": {"color_enabled": True}},
        {"key": "color_max_shift", "label": "Max colour shift (Lab units)", "type": "range", "min": 1.0, "max": 20.0, "step": 0.5, "show_if": {"color_enabled": True}},
        {"key": "color_match_std", "label": "Also match contrast (std-dev)", "type": "bool", "show_if": {"color_enabled": True}},
        {"key": "protect_faces", "label": "Protect faces (OpenCV frontal-face detector)", "type": "bool", "show_if": {"color_enabled": True}},
        {"key": "protect_skin", "label": "Protect skin tones", "type": "range", "min": 0.0, "max": 1.0, "step": 0.05, "show_if": {"color_enabled": True}},
    ]},
]

BUILTIN_PRESETS = {
    "identity_safe": {"label": "Faces & identity safe (recommended)", "settings": {"mode": "refine", "restore_first": False, "resolution_mode": "scale", "scale": 1.5, "guide_strength": 1.0, "color_enabled": True, "color_auto": False, "color_mode": "source", "color_strength": 0.85, "color_match_std": True, "protect_faces": True, "decode_tile": 1024}},
    "detail_4k": {"label": "4K detail, single pass", "settings": {"mode": "refine", "restore_first": False, "resolution_mode": "preset", "preset_long_edge": 3840, "guide_strength": 1.0, "cached_cond": True, "color_enabled": True, "color_auto": False, "color_mode": "source", "color_strength": 0.85, "color_match_std": True, "decode_tile": 1024}},
    "max_texture": {"label": "Max texture (Pixel, fast)", "settings": {"mode": "pixel", "restore_first": False, "resolution_mode": "scale", "scale": 2.0, "color_enabled": True, "color_auto": False, "color_mode": "source", "color_strength": 0.85, "color_match_std": True, "decode_tile": 1024}},
    "old_footage": {"label": "Old footage: Restore then Refine", "settings": {"mode": "refine", "restore_first": True, "resolution_mode": "scale", "scale": 1.5, "guide_strength": 1.0, "color_enabled": True, "color_auto": False, "color_mode": "temporal", "color_strength": 0.85, "decode_tile": 1024}},
    "low_vram": {"label": "Low VRAM (GGUF)", "settings": {"mode": "refine", "restore_first": False, "precision": "gguf", "cached_cond": True, "decode_tile": 256, "decode_temporal": 32, "budget_fmp": 250, "color_enabled": True, "color_auto": False, "color_mode": "source", "color_strength": 0.85}},
    "in_place": {"label": "Re-detail in place (1.0x)", "settings": {"mode": "refine", "restore_first": False, "resolution_mode": "scale", "scale": 1.0, "guide_strength": 1.0, "color_enabled": True, "color_auto": False, "color_mode": "source", "color_strength": 0.85, "color_match_std": True, "decode_tile": 1024}},
}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def normalize(raw: dict):
    """Return (clean_settings, warnings). Unknown keys are dropped; values are coerced and clamped."""
    s = copy.deepcopy(DEFAULTS)
    warn = []
    raw = raw or {}
    for k, default in DEFAULTS.items():
        if k not in raw or raw[k] is None and default is not None:
            continue
        v = raw[k]
        try:
            if isinstance(default, bool):
                v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
            elif isinstance(default, int) and not isinstance(default, bool):
                v = int(float(v))
            elif isinstance(default, float):
                v = float(v)
            elif default is None:
                v = None if v in (None, "") else float(v)
            else:
                v = str(v)
        except (TypeError, ValueError):
            warn.append(f"{k}: invalid value {raw[k]!r}, using default")
            continue
        if k in OPTIONS and v not in OPTIONS[k]:
            warn.append(f"{k}: {v!r} not allowed, using {default!r}")
            continue
        s[k] = v
    s["scale"] = _clamp(s["scale"], 1.0, 2.0)
    if not s.get("sampler_name") or s["sampler_name"] not in OPTIONS["sampler_name"]:
        s["sampler_name"] = "euler"
    s["guide_strength"] = _clamp(s["guide_strength"], 0.3, 1.0)
    s["lora_strength"] = _clamp(s["lora_strength"], 0.0, 1.5)
    s["crf"] = int(_clamp(s["crf"], 8, 28))
    s["color_strength"] = _clamp(s["color_strength"], 0.0, 1.0)
    s["protect_skin"] = _clamp(s["protect_skin"], 0.0, 1.0)
    s["color_window"] = int(_clamp(s["color_window"], 1, 61)) | 1
    s["color_max_shift"] = _clamp(s["color_max_shift"], 1.0, 20.0)
    s["overlap_frac"] = _clamp(s["overlap_frac"], 0.0, 0.5)
    s["blend_var"] = _clamp(s["blend_var"], 0.0, 0.5)
    s["grid_cycle"] = int(_clamp(s["grid_cycle"], 1, 8))
    s["tile_frames"] = int(_clamp(s["tile_frames"], 0, 241))
    s["tile_w"] = int(_clamp(s["tile_w"], 256, 2048)) // 32 * 32
    s["tile_h"] = int(_clamp(s["tile_h"], 256, 2048)) // 32 * 32
    s["decode_tile"] = int(_clamp(s["decode_tile"], 0, 1024))
    s["decode_temporal"] = int(_clamp(s["decode_temporal"], 0, 256))
    s["budget_fmp"] = float(_clamp(s["budget_fmp"], 0, 2000))
    s["max_chunk_s"] = float(_clamp(s["max_chunk_s"], 0, 60))
    s["seed"] = int(_clamp(s["seed"], 0, 2**31 - 1))
    if s["mode"] == "pixel":
        if s["restore_first"]:
            warn.append("restore_first only applies to Refine; ignored in Pixel mode")
            s["restore_first"] = False
        if s["guide_strength"] != 1.0:
            s["guide_strength"] = 1.0
    if s["mode"] == "restore" and s["restore_first"]:
        s["restore_first"] = False
    if s["mode"] != "pixel" and (s["prompt_positive"].strip() or s["prompt_negative"].strip()):
        if s["cached_cond"]:
            warn.append("Custom prompts need the text encoder: cached conditioning was switched off for this job")
        s["cached_cond"] = False
    if s["mode"] == "pixel" and (s["prompt_positive"].strip() or s["prompt_negative"].strip()):
        warn.append("Pixel mode uses empty prompts; the prompt fields are ignored")
        s["prompt_positive"] = s["prompt_negative"] = ""
    if s["guide_strength"] < 0.6 and s["mode"] != "pixel":
        warn.append("Guide strength below 0.6 trades fidelity for texture; faces may drift to a different person")
    if s["sigmas_mode"] == "custom":
        try:
            vals = [float(x) for x in re.split(r"[,\s]+", s["sigmas_custom"].strip()) if x]
            if len(vals) < 3 or vals[-1] != 0.0 or any(b >= a for a, b in zip(vals, vals[1:])) or vals[0] > 1.0:
                raise ValueError
            s["sigmas_custom"] = ", ".join(repr(v) for v in vals)
        except ValueError:
            warn.append("Custom sigmas are invalid (need >=3 strictly descending values in (0,1] ending at 0); using the distilled schedule")
            s["sigmas_mode"] = "distilled"
    if s["vcodec"] == "libx265" and s["container"] == "mov":
        s["container"] = "mp4"
        warn.append("HEVC in mov is not widely supported; using mp4")
    if s["test_enabled"]:
        s["test_frames"] = s["test_frames"] if s["test_frames"] % 8 == 1 else (s["test_frames"] // 8) * 8 + 1
    return s, warn


def graph_hash_keys():
    """Settings that change what a rendered chunk looks like (used for resume hashing)."""
    return ["mode", "restore_first", "resolution_mode", "scale", "preset_long_edge", "guide_strength", "lora_strength", "seed",
            "precision", "gguf_file", "cached_cond", "encoder", "clip_device", "prompt_positive", "prompt_negative",
            "sampler_name", "sigmas_mode", "sigmas_custom", "tile_mode", "tile_w", "tile_h", "overlap_frac", "blend_var",
            "grid_cycle", "tile_frames", "canvas_device", "video_vae", "audio_vae", "decode_tile", "decode_temporal"]
