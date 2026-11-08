"""Constructs and customizes the ReDetail 2.0 API graphs for ComfyUI.
Supports Refine Details, Restore, and Pixel Spatial Upscaler, with INT8 ConvRot, GGUF,
Cached Conditioning, Custom Sigmas, Tiled Fusion parameters, and VAE decode tuning."""
import copy, json, os
from pathlib import Path

from common import REDETAIL, ram_limits

WF_REFINE_PATH = REDETAIL / "workflows" / "ltx25_refine_API.json"
WF_PIXEL_PATH = REDETAIL / "workflows" / "ltx25_upscale_API.json"

# Node IDs in the verified ReDetail API workflows
N_FRAME, N_VIDEO, N_SAVE = "2004", "5001", "4852"
N_REF_RESIZE, N_CANVAS = "5548:5026", "9002:3059"
N_GUIDE_RESIZE, N_GUIDE, N_SAMPLER = "9002:9201", "9002:5012", "5516:9100"


def prune_unreachable(graph: dict, root_node_id: str = N_SAVE) -> dict:
    """Prunes dead or disconnected branches from ComfyUI graph to avoid validation failures."""
    keep = set()
    stack = [root_node_id]
    while stack:
        nid = stack.pop()
        if nid in keep or nid not in graph:
            continue
        keep.add(nid)
        for val in graph[nid].get("inputs", {}).values():
            if isinstance(val, list) and len(val) > 0:
                stack.append(str(val[0]))
    for dead in list(graph.keys()):
        if dead not in keep:
            del graph[dead]
    return graph


def apply_cached_conditioning(pr: dict, name: str, out_node: str = N_SAVE) -> dict:
    """Replaces CLIPTextEncode branches with pre-computed cached conditioning tensor loader."""
    pr["5014:2483"] = {"class_type": "ReDetailLoadConditioning", "inputs": {"name": f"{name}_pos"}}
    pr["5014:2612"] = {"class_type": "ReDetailLoadConditioning", "inputs": {"name": f"{name}_neg"}}
    if "9002:9005" in pr:
        pr["9002:9005"]["inputs"]["positive"] = ["5014:1241", 0]
        pr["9002:9005"]["inputs"]["negative"] = ["5014:1241", 1]
    return prune_unreachable(pr, out_node)


def build_workflow(settings: dict, input_video_name: str, output_prefix: str, tw: int, th: int, cw: int, ch: int, save_options: dict = None, first_frame_image_name: str = None) -> dict:
    mode = settings.get("mode", "refine")
    is_pixel = (mode == "pixel")
    wf_path = WF_PIXEL_PATH if is_pixel else WF_REFINE_PATH

    if not wf_path.exists():
        raise FileNotFoundError(f"Workflow template missing: {wf_path}")

    base = json.loads(wf_path.read_text())
    pr = copy.deepcopy(base)

    # 1. Inputs and Output prefixes
    pr[N_VIDEO]["inputs"]["file"] = input_video_name
    pr[N_SAVE]["inputs"]["filename_prefix"] = output_prefix
    if save_options:
        pr[N_SAVE]["inputs"].update(save_options)

    # 2. Target Resolution & Canvas
    pr[N_CANVAS]["inputs"]["width"] = tw
    pr[N_CANVAS]["inputs"]["height"] = th

    # 3. Mode-specific guide, tiles & samplers
    if not is_pixel:
        # Refine or Restore
        lora_file = ("ltx-2.5-22b-ic-lora-restore-1.0.safetensors"
                     if mode == "restore"
                     else "ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors")
        if "5004:5606" in pr:
            pr["5004:5606"]["inputs"]["lora_name"] = lora_file
            pr["5004:5606"]["inputs"]["strength_model"] = float(settings.get("lora_strength", 1.0))

        # Guide image scaling to solved canvas
        if N_GUIDE_RESIZE in pr:
            pr[N_GUIDE_RESIZE]["inputs"]["width"] = tw
            pr[N_GUIDE_RESIZE]["inputs"]["height"] = th

        # Guide strength
        if N_GUIDE in pr:
            pr[N_GUIDE]["inputs"]["strength"] = float(settings.get("guide_strength", 1.0))

        # Tiled Fusion Sampler
        if N_SAMPLER in pr:
            tile_mode = settings.get("tile_mode", "auto")
            if tile_mode == "manual":
                tile_w = int(settings.get("tile_w", 1024))
                tile_h = int(settings.get("tile_h", 576))
            else:
                # Auto: Use trained dimensions (landscape: 1024x576, portrait: 576x1024)
                tile_w, tile_h = (576, 1024) if th > tw else (1024, 576)

            pr[N_SAMPLER]["inputs"]["tile_width"] = tile_w
            pr[N_SAMPLER]["inputs"]["tile_height"] = tile_h
            pr[N_SAMPLER]["inputs"]["seed"] = int(settings.get("seed", 42))
            pr[N_SAMPLER]["inputs"]["cfg"] = 1.0
            pr[N_SAMPLER]["inputs"]["blend_var"] = float(settings.get("blend_var", 0.05))
            pr[N_SAMPLER]["inputs"]["grid_cycle"] = int(settings.get("grid_cycle", 1))

            ov = float(settings.get("overlap_frac", 0.0))
            if ov > 0:
                pr[N_SAMPLER]["inputs"]["overlap_frac"] = ov
            tf = int(settings.get("tile_frames", 0))
            if tf > 0:
                pr[N_SAMPLER]["inputs"]["tile_frames"] = tf
            cd = settings.get("canvas_device", "auto")
            if cd in ("auto", "cpu", "gpu"):
                pr[N_SAMPLER]["inputs"]["canvas_device"] = cd
    else:
        # Pixel Spatial Upscaler
        if first_frame_image_name and N_FRAME in pr:
            pr[N_FRAME]["inputs"]["image"] = first_frame_image_name
        if N_REF_RESIZE in pr:
            pr[N_REF_RESIZE]["inputs"]["resize_type.shorter_size"] = min(cw, ch)
        if "5004:5606" in pr:
            pr["5004:5606"]["inputs"]["strength_model"] = float(settings.get("lora_strength", 1.0))
        if "5516:4832" in pr:
            pr["5516:4832"]["inputs"]["noise_seed"] = int(settings.get("seed", 42))

    # 4. Sampler name and Sigmas
    s_name = (settings.get("sampler_name") or "euler").strip() or "euler"
    if "5516:4831" in pr:
        pr["5516:4831"]["inputs"]["sampler_name"] = s_name
    if settings.get("sigmas_mode") == "custom" and settings.get("sigmas_custom"):
        if "5516:4984" in pr:
            pr["5516:4984"]["inputs"]["sigmas"] = settings["sigmas_custom"]

    # 5. Model Precision (INT8 ConvRot vs GGUF) & VAE selection
    v_vae = (settings.get("video_vae") or "").strip()
    a_vae = (settings.get("audio_vae") or "").strip()
    if v_vae and "5004:5601" in pr and "inputs" in pr["5004:5601"]:
        pr["5004:5601"]["inputs"]["vae_name"] = v_vae
    if a_vae and "5004:5600" in pr and "inputs" in pr["5004:5600"]:
        pr["5004:5600"]["inputs"]["vae_name"] = a_vae

    precision = settings.get("precision", "int8")
    if precision == "gguf":
        gguf_name = settings.get("gguf_file", "LTX-2.5-Distilled-Q4_K_M.gguf")
        pr["5004:5602"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": gguf_name}}
        for nid in ("5014:5504", "5014:5505"):
            if nid in pr and "ckpt_name" in pr[nid]["inputs"]:
                pr[nid]["inputs"]["ckpt_name"] = gguf_name

    # 6. Prompts & Conditioning
    has_custom_prompts = bool((settings.get("prompt_positive") or "").strip() or (settings.get("prompt_negative") or "").strip())
    use_cache = settings.get("cached_cond", True) and (is_pixel or not has_custom_prompts)

    if use_cache:
        cond_name = "redetail" if is_pixel else "redetail_refine"
        pr = apply_cached_conditioning(pr, cond_name, N_SAVE)
    else:
        # Custom or standard text encoder
        enc = settings.get("encoder")
        dev = settings.get("clip_device")
        for nid in ("5004:5604", "5004:5605"):
            if nid in pr:
                if enc:
                    pr[nid]["inputs"]["clip_name"] = enc
                if dev and "device" in pr[nid]["inputs"]:
                    pr[nid]["inputs"]["device"] = dev
        if has_custom_prompts and not is_pixel:
            pos_p = settings.get("prompt_positive", "").strip()
            neg_p = settings.get("prompt_negative", "").strip()
            if pos_p and "5508" in pr:
                pr["5508"]["inputs"]["value"] = pos_p
            if neg_p and "5509" in pr:
                pr["5509"]["inputs"]["value"] = neg_p

    # 7. Ensure CreateVideo explicitly sets sRGB (BT.709 matrix) across all modes
    cv_node = pr.get("5518:4849")
    if cv_node and "inputs" in cv_node:
        cv_node["inputs"].setdefault("color_space", "sRGB")
        cv_node["inputs"].setdefault("codec", "none")

    # 8. VAE Decode Tiled settings
    # Note: Diffusion sampling already uses 1024x576 tiles (LTXVTiledFusionSampler).
    # VAEDecodeTiled is a separate memory spike after sampling completes; on <64 GB system RAM
    # or >=2.5K canvases, capping tile_size=512 (with generous 128px overlap) and temporal_size=64
    # prevents Linux cgroup OOM kills during VAE decode while keeping tile transitions seamless.
    dec_tile = int(settings.get("decode_tile", 0))
    dec_temp = int(settings.get("decode_temporal", 0))
    vaedec_node = pr.get("5518:5538")
    if vaedec_node and "inputs" in vaedec_node:
        total_ram_gb, _ = ram_limits()
        eff_tile = dec_tile if dec_tile > 0 else 512
        if total_ram_gb < 64 and tw * th >= 2560 * 1440 and eff_tile > 512:
            eff_tile = 512
        vaedec_node["inputs"]["tile_size"] = eff_tile
        vaedec_node["inputs"]["overlap"] = 128 if eff_tile >= 512 else max(32, (eff_tile // 4) // 32 * 32)
        eff_temp = dec_temp if dec_temp > 0 else (64 if total_ram_gb < 64 else 128)
        vaedec_node["inputs"]["temporal_size"] = eff_temp
        vaedec_node["inputs"]["temporal_overlap"] = min(32, max(16, (eff_temp // 4) // 8 * 8))

    return pr
