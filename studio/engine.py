"""LTX Video Enhancement Engine: Smart Chunking, Pre-flight validation,
ComfyUI execution, Assembly, Audio preservation, and Color consistency."""
import asyncio, copy, json, math, os, shutil, subprocess, time, uuid
from pathlib import Path

import aiohttp
import psutil

from common import ROOT, DIRS, load_config, atomic_write_json, read_json, fmt_eta
from video import probe_video, detect_scene_cuts, target_resolution, smart_chunking, render_length_8n1
from graph import build_workflow
from comfy import ComfyClient, PromptRejected
import colorfix
import sysmon


class JobCancelled(Exception):
    pass


class Engine:
    def __init__(self, comfy_client: ComfyClient = None):
        self.comfy = comfy_client or ComfyClient()
        self._idle_free_task = None
        self._execution_lock = asyncio.Lock()
        self._idle_generation = 0

    async def free_h3_cache(self):
        """Politely requests H3 ComfyUI on port 8188 to unload cached models and free PyTorch VRAM."""
        cfg = load_config()
        if not cfg.get("coordinate_h3", True):
            return
        h3_url = cfg.get("h3_url", "http://127.0.0.1:8188").rstrip("/")
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(f"{h3_url}/queue", timeout=aiohttp.ClientTimeout(total=4)) as r:
                    if r.status != 200:
                        return
                    queue = await r.json()
                if not isinstance(queue, dict) or any(not isinstance(queue.get(k), list) for k in ("queue_running", "queue_pending")):
                    return
                if queue["queue_running"] or queue["queue_pending"]:
                    return
                await s.post(f"{h3_url}/free", json={"unload_models": True, "free_memory": True}, timeout=aiohttp.ClientTimeout(total=4))
        except Exception:
            pass

    async def check_h3_busy(self) -> bool:
        """Multi-tiered check: verifies if H3 Studio on port 8188 is actively running, decoding, encoding, or holding GPU memory."""
        cfg = load_config()
        if not cfg.get("coordinate_h3", True):
            return False
        h3_url = cfg.get("h3_url", "http://127.0.0.1:8188").rstrip("/")

        # Tier 1: Check ComfyUI 8188 HTTP Queue
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(f"{h3_url}/queue", timeout=aiohttp.ClientTimeout(total=8)) as r:
                    if r.status == 200:
                        data = await r.json()
                        running = data.get("queue_running", [])
                        pending = data.get("queue_pending", [])
                        if len(running) > 0 or len(pending) > 0:
                            return True
                    else:
                        # Non-200 means server is restarting, erroring, or locked
                        return True
        except (asyncio.TimeoutError, aiohttp.ServerTimeoutError):
            # When H3 is under heavy compute/decoding, its event loop is sluggish -> assume busy!
            return True
        except Exception:
            # Unreachable / connection refused: H3 server may be down or starting
            pass

        # Tier 2: Check for active encoding / ffmpeg processes under H3 or ComfyUI
        try:
            for p in psutil.process_iter(["pid", "name", "cmdline"]):
                name = p.info.get("name", "").lower()
                if "ffmpeg" in name or "ffprobe" in name:
                    cmd = " ".join(p.info.get("cmdline") or [])
                    if "ComfyUI" in cmd or "h3" in cmd or "output" in cmd:
                        return True
        except Exception:
            pass

        # Tier 3: Physical GPU VRAM and Process Memory Guard
        try:
            g = sysmon.gpu_info()
            if g.get("available") and g.get("vram_total_mb"):
                free_mb = g["vram_total_mb"] - g.get("vram_used_mb", 0)

                # Identify LTX's own worker PID
                ltx_worker_pid = None
                p2 = DIRS["state"] / "comfy.pid"
                if p2.exists():
                    try:
                        ltx_worker_pid = json.loads(p2.read_text()).get("pid")
                    except Exception:
                        pass

                # Sum VRAM of all other processes on the GPU (e.g. H3 PID 131001)
                own_vram = 0
                other_procs_vram = 0
                for proc in g.get("procs", []):
                    if proc["pid"] != ltx_worker_pid:
                        other_procs_vram += proc.get("mb", 0)
                    else:
                        own_vram += proc.get("mb", 0)

                # If H3 or other apps are holding > 4GB VRAM, H3 is still resident/caching/decoding
                if other_procs_vram > 4000:
                    return True

                # If total free VRAM is under 20 GB, there's not enough room to safely run 22B LTX
                if free_mb + own_vram < 20000:
                    return True
        except Exception:
            pass

        return False

    async def release_temporary_memory(self):
        # /free with free_memory=True resets ComfyUI's executor cache. RAM-pressure
        # caching already evicts tensors at the configured headroom thresholds;
        # resetting it every chunk defeats loader and conditioning reuse.
        cfg = load_config()
        adaptive = cfg.get("cache_mode") == "ram" or "--cache-ram" in cfg.get("comfy_extra_args", [])
        if not adaptive:
            await self.comfy.free(unload=False, free_memory=True)

    def reusable_vram_mb(self):
        """Free memory plus this worker's reusable model cache; never count other workers."""
        gpu = sysmon.gpu_info()
        if not gpu.get("available") or not gpu.get("vram_total_mb"):
            return None
        own_pid = read_json(DIRS["state"] / "comfy.pid", {}).get("pid")
        own_mb = sum(p.get("mb", 0) for p in gpu.get("procs", [])
                     if own_pid is not None and p.get("pid") == own_pid)
        return gpu["vram_total_mb"] - gpu.get("vram_used_mb", 0) + own_mb

    async def wait_for_gpu_vram(self, job: dict, on_update):
        """Coordinates GPU access: if H3 Studio is actively rendering, decoding, or encoding, waits until GPU frees up."""
        waited = 0
        max_wait = 7200  # up to 2 hours
        while waited < max_wait:
            if job.get("status") == "cancelled":
                raise JobCancelled()
            # An idle H3 worker may still own its cached model. Release that cache
            # before applying the physical-memory guard, otherwise admission stalls.
            await self.free_h3_cache()
            h3_busy = await self.check_h3_busy()
            free_mb = self.reusable_vram_mb()
            if not h3_busy and (free_mb is None or free_mb >= 20000):
                # Settle cooldown: wait 6 seconds to ensure any final encoding / file writes have fully settled
                await asyncio.sleep(6)
                # Request H3 to release unused PyTorch cached VRAM
                await self.free_h3_cache()
                await asyncio.sleep(2)
                # Final check after settle
                h3_busy_again = await self.check_h3_busy()
                free_mb_again = self.reusable_vram_mb()
                if not h3_busy_again and (free_mb_again is None or free_mb_again >= 20000):
                    break

            msg = f"Waiting for GPU (H3 Studio busy/encoding or <20GB VRAM free: {free_mb:.0f} MB)..." if free_mb else "Waiting for GPU (H3 Studio active)..."
            job["stage"] = msg
            job["eta_text"] = "Queued (Waiting for GPU)"
            on_update(job)
            await asyncio.sleep(5)
            waited += 5

    def preflight_models(self, settings: dict):
        """Verifies all required models exist on disk before submitting prompt."""
        mdir = DIRS["models"]
        precision = settings.get("precision", "int8")
        mode = settings.get("mode", "refine")
        cached = settings.get("cached_cond", True)

        required = [
            ("Video VAE", mdir / "vae" / settings.get("video_vae", "ltx-2.5-video-vae-bf16.safetensors")),
            ("Audio VAE", mdir / "vae" / settings.get("audio_vae", "ltx-2.5-audio-vae-bf16.safetensors")),
        ]

        if precision == "gguf":
            required.append(("GGUF Transformer", mdir / "unet" / settings.get("gguf_file", "LTX-2.5-Distilled-Q4_K_M.gguf")))
        else:
            required.append(("DiT INT8 Transformer", mdir / "diffusion_models" / "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"))

        if not cached and not (mode == "pixel"):
            enc = settings.get("encoder") or "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
            required.append(("Text Encoder", mdir / "text_encoders" / enc))

        if mode == "pixel":
            required.append(("Pixel IC-LoRA", mdir / "loras" / "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"))
        elif mode == "restore":
            required.append(("Restore IC-LoRA", mdir / "loras" / "ltx-2.5-22b-ic-lora-restore-1.0.safetensors"))
        else:
            required.append(("Refine IC-LoRA", mdir / "loras" / "ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors"))

        missing = []
        for label, path in required:
            if not path.exists() or path.stat().st_size < 1000:
                missing.append(f"{label} ({path.name})")

        if missing:
            raise RuntimeError(f"Missing required models: {', '.join(missing)}. Please check the installer progress or download status.")

    async def execute(self, job: dict, on_update):
        # Serialize admission against an idle unload already in flight. Do not cancel
        # an accepted /free request: wait for the idle critical section to finish.
        self._idle_generation += 1
        async with self._execution_lock:
            if self._idle_free_task is not None:
                self._idle_free_task.cancel()
                try:
                    await self._idle_free_task
                except asyncio.CancelledError:
                    pass
                self._idle_free_task = None
            return await self._execute(job, on_update)

    async def _execute(self, job: dict, on_update):
        job_id = job["id"]
        src = job["input_video_path"]
        settings = job["settings"]
        work_dir = DIRS["jobs"] / f"work_{job_id}"
        work_dir.mkdir(parents=True, exist_ok=True)
        job["work_dir"] = str(work_dir)

        t_start = time.time()
        job["status"] = "running"
        job["stage"] = "Pre-flight validation..."
        job["progress"] = 2.0
        on_update(job)

        # 1. Validation
        self.preflight_models(settings)
        await self.wait_for_gpu_vram(job, on_update)

        # 2. Probe source video
        job["stage"] = "Analyzing input video streams & cuts..."
        info = probe_video(src)
        job["video_info"] = info
        dur = info["duration"]
        fps = info["fps"]
        total_frames = info["frames"]

        # Range clip selection if requested
        r_start = settings.get("range_start_s")
        r_end = settings.get("range_end_s")
        if r_start is not None or r_end is not None:
            r_s = max(0.0, float(r_start or 0.0))
            r_e = min(dur, float(r_end or dur))
            if r_e > r_s + 0.5:
                dur = r_e - r_s
                total_frames = round(dur * fps)

        # 3. Calculate target geometry
        mode = settings.get("mode", "refine")
        res_mode = settings.get("resolution_mode", "scale")
        scale = float(settings.get("scale", 1.5))
        long_edge = int(settings.get("preset_long_edge", 1920))
        geom = target_resolution(info["width"], info["height"], mode, res_mode, scale, long_edge)
        job["geometry"] = geom
        tw, th, cw, ch = geom["tw"], geom["th"], geom["cw"], geom["ch"]

        # Test render mode vs full video
        is_test = bool(settings.get("test_enabled", False))
        if is_test:
            test_f = int(settings.get("test_frames", 17))
            test_f = render_length_8n1(test_f)
            t_s = float(settings.get("test_start_s", 0.0))
            chunks = [{
                "index": 0, "start_frame": round(t_s * fps), "end_frame": round(t_s * fps) + test_f,
                "frames": test_f, "start_s": t_s, "end_s": t_s + (test_f / fps),
                "duration_s": test_f / fps, "render_frames": test_f + (8 if mode != "pixel" else 0)
            }]
            cuts = []
        else:
            job["stage"] = "Detecting scene cuts for smart chunking..."
            on_update(job)
            cuts = detect_scene_cuts(src)
            job["scene_cuts"] = cuts

            # Memory budget: if 0, auto-derive from RAM/VRAM
            budget = float(settings.get("budget_fmp", 0.0))
            if budget <= 0:
                budget = 850.0 if mode == "pixel" else 500.0
            max_c_s = float(settings.get("max_chunk_s", 0.0))
            chunks = smart_chunking(total_frames, fps, cuts, budget, tw, th, refine=(mode != "pixel"), max_chunk_s=max_c_s)

        job["chunks"] = chunks
        job["total_chunks"] = len(chunks)
        job["stage"] = f"Prepared {len(chunks)} chunk(s). Starting render..."
        job["progress"] = 5.0
        on_update(job)

        # 4. Process each chunk
        rendered_parts = []
        save_opts = await self.comfy.save_options(crf=int(settings.get("crf", 14)))

        for idx, chunk in enumerate(chunks):
            if job.get("status") == "cancelled":
                raise JobCancelled()

            c_tag = f"ltx_{job_id}_c{idx:03d}"
            c_input = work_dir / f"in_{c_tag}.mp4"
            c_frames = chunk["frames"]
            r_len = chunk["render_frames"]
            s_time = chunk["start_s"]

            job["stage"] = f"Rendering chunk {idx + 1}/{len(chunks)} ({c_frames} frames)..."
            job["current_chunk"] = idx + 1
            on_update(job)

            # Extract chunk video with ffmpeg
            vf = f"scale={cw}:{ch}:flags=lanczos" if geom["fit"] == "resize" else (f"crop={cw}:{ch}" if geom["fit"] == "crop" else "null")
            pad = f",tpad=stop_mode=clone:stop_duration={r_len / fps:.3f}" if (idx == len(chunks) - 1 or s_time + (r_len / fps) > dur) else ""
            cut_cmd = [
                "ffmpeg", "-y", "-v", "error", "-ss", f"{max(0.0, s_time - 0.5/fps):.6f}",
                "-i", src
            ]
            if not info["has_audio"]:
                cut_cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
            cut_cmd += [
                "-frames:v", str(r_len), "-vf", vf + pad,
                "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k"
            ]
            if not info["has_audio"]:
                cut_cmd += ["-map", "0:v:0", "-map", "1:a:0", "-shortest"]
            cut_cmd += [str(c_input)]
            subprocess.run(cut_cmd, check=True)

            # Upload chunk to ComfyUI
            up_name = await self.comfy.upload(str(c_input))

            first_frame_name = None
            if mode == "pixel":
                frm_path = work_dir / f"f0_{c_tag}.png"
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(c_input), "-frames:v", "1", "-update", "1", str(frm_path)], check=True)
                first_frame_name = await self.comfy.upload(str(frm_path))

            # Flush unused cached tensors before each chunk to prevent OOM
            await self.release_temporary_memory()

            # Build and submit workflow
            prompt = build_workflow(settings, up_name, c_tag, tw, th, cw, ch, save_options=save_opts, first_frame_image_name=first_frame_name)
            prompt_id = await self.comfy.submit(prompt, extra={"job_id": job_id, "chunk": idx})
            job["prompt_id"] = prompt_id

            # Monitor progress via WS
            stop_ev = asyncio.Event()

            def on_ws(evt_type, data):
                if evt_type == "progress":
                    val = data.get("value", 0)
                    m = data.get("max", 8)
                    job["step"] = val
                    job["max_steps"] = m
                    chunk_frac = (idx + (val / max(1, m))) / len(chunks)
                    job["progress"] = round(5.0 + chunk_frac * 80.0, 1)
                    elapsed = time.time() - t_start
                    if chunk_frac > 0:
                        total_est = elapsed / chunk_frac
                        rem = max(0.0, total_est - elapsed)
                        job["eta_text"] = fmt_eta(rem)
                        job["eta_seconds"] = round(rem)
                    on_update(job)
                elif evt_type == "execution_error":
                    job["error"] = data.get("exception_message")

            ws_task = asyncio.create_task(self.comfy.watch(prompt_id, on_ws, stop_ev))

            # Poll history until finished with detection for worker restart
            history_item = None
            not_found_ticks = 0
            while True:
                if job.get("status") == "cancelled":
                    await self.comfy.interrupt()
                    stop_ev.set()
                    ws_task.cancel()
                    raise JobCancelled()
                try:
                    h = await self.comfy.history(prompt_id)
                    if prompt_id in h:
                        st = h[prompt_id].get("status", {})
                        if st.get("status_str") == "success":
                            out = h[prompt_id].get("outputs", {}).get("4852", {})
                            for k in ("video", "videos", "gifs", "images"):
                                for itm in out.get(k) or []:
                                    if itm.get("type", "output") == "output":
                                        history_item = itm
                                        break
                            break
                        else:
                            msgs = st.get("messages", [])
                            err_txt = "ComfyUI execution failed"
                            for m in msgs:
                                if len(m) > 1 and isinstance(m[1], dict):
                                    if "exception_message" in m[1]:
                                        err_txt = m[1]["exception_message"]
                                    elif "node_errors" in m[1]:
                                        err_txt = "Node error: " + str(m[1]["node_errors"])
                            raise RuntimeError(err_txt)
                    else:
                        # Check whether prompt is still queued or active in ComfyUI
                        try:
                            q = await self.comfy.queue()
                            running = any(item[1] == prompt_id for item in q.get("queue_running", []))
                            pending = any(item[1] == prompt_id for item in q.get("queue_pending", []))
                            if not running and not pending:
                                not_found_ticks += 1
                                if not_found_ticks >= 6:
                                    raise RuntimeError("ComfyUI worker restarted or prompt lost from queue")
                            else:
                                not_found_ticks = 0
                        except Exception as q_err:
                            if "restarted" in str(q_err):
                                raise
                except (PromptRejected, RuntimeError):
                    raise
                except Exception:
                    pass
                await asyncio.sleep(2)

            stop_ev.set()
            ws_task.cancel()

            # Flush memory immediately after chunk completion
            await self.release_temporary_memory()

            if not history_item:
                raise RuntimeError(f"Chunk {idx} produced no output file")

            # Download chunk output
            dst_chunk = work_dir / f"out_{c_tag}.mp4"
            await self.comfy.download(history_item, str(dst_chunk))

            # Trim extra 8 frames if refine padding was applied
            actual_f = int(subprocess.check_output([
                "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(dst_chunk)
            ], text=True).strip() or "0")

            if actual_f > c_frames:
                trimmed_chunk = work_dir / f"t_{c_tag}.mp4"
                subprocess.run([
                    "ffmpeg", "-y", "-v", "error", "-i", str(dst_chunk),
                    "-frames:v", str(c_frames), "-c:v", "libx264", "-crf", "14",
                    "-pix_fmt", "yuv420p", "-c:a", "copy", str(trimmed_chunk)
                ], check=True)
                os.replace(trimmed_chunk, dst_chunk)

            rendered_parts.append(str(dst_chunk))

        # 5. Re-time and Concat
        job["stage"] = "Concatenating chunks and verifying frame synchronization..."
        job["progress"] = 87.0
        on_update(job)

        retimed_parts = []
        for p in rendered_parts:
            rt = work_dir / f"rt_{os.path.basename(p)}"
            subprocess.run([
                "ffmpeg", "-y", "-v", "error", "-i", p,
                "-vf", f"setpts=N/(({fps})*TB)", "-fps_mode", "passthrough",
                "-c:v", "libx264", "-crf", "15", "-pix_fmt", "yuv420p", "-an",
                str(rt)
            ], check=True)
            retimed_parts.append(str(rt))

        concat_list = work_dir / "concat.txt"
        concat_list.write_text("".join(f"file '{p}'\n" for p in retimed_parts))
        joined_raw = work_dir / "joined_raw.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
            "-i", str(concat_list), "-c:v", "libx264", "-crf", "15",
            "-fps_mode", "passthrough", "-pix_fmt", "yuv420p", "-an",
            str(joined_raw)
        ], check=True)

        # 6. Color Consistency pass if enabled
        assembled_video = joined_raw
        color_report = None
        if settings.get("color_enabled", False):
            job["stage"] = "Analyzing color consistency & temporal flicker..."
            job["progress"] = 90.0
            on_update(job)

            # Sample low-res stats for both source and assembled
            w_stat, h_stat = 320, int(round(320 * (th / tw)))
            src_cmd = colorfix.lowres_cmd(["-i", src], f"scale={tw}:{th}:flags=lanczos", w_stat, h_stat)
            out_cmd = colorfix.lowres_cmd(["-i", str(joined_raw)], "", w_stat, h_stat)

            bounds_frames = [c["start_frame"] for c in chunks[1:]]
            src_stats = colorfix.collect_stats(src_cmd, w_stat, h_stat)
            out_stats = colorfix.collect_stats(out_cmd, w_stat, h_stat)

            color_report = colorfix.analyze(src_stats, out_stats, fps, cuts, bounds_frames)
            job["color_analysis"] = color_report

            should_apply = True
            if settings.get("color_auto", True) and color_report["verdict"] == "ok":
                should_apply = False

            if should_apply:
                job["stage"] = "Applying color matching & flicker correction (face protected)..."
                on_update(job)
                plan_ = colorfix.plan(
                    src_stats, out_stats, cuts, bounds_frames,
                    mode=settings.get("color_mode", "source"),
                    strength=float(settings.get("color_strength", 0.7)),
                    window=int(settings.get("color_window", 9)),
                    max_shift=float(settings.get("color_max_shift", 6.0)),
                    match_std=bool(settings.get("color_match_std", False))
                )
                corrected_video = work_dir / "color_corrected.mp4"
                dec_c = ["ffmpeg", "-v", "error", "-i", str(joined_raw), "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
                enc_c = [
                    "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{tw}x{th}", "-r", str(fps), "-i", "-",
                    "-c:v", settings.get("vcodec", "libx264"), "-crf", str(settings.get("crf", 14)),
                    "-preset", settings.get("x_preset", "medium"), "-pix_fmt", settings.get("pix_fmt", "yuv420p"),
                    str(corrected_video)
                ]
                colorfix.apply(
                    dec_c, enc_c, tw, th, plan_,
                    protect_faces=bool(settings.get("protect_faces", True)),
                    protect_skin=float(settings.get("protect_skin", 0.6))
                )
                assembled_video = corrected_video

        # 7. Audio Mux & Final Export
        job["stage"] = "Muxing audio losslessly and finalizing container..."
        job["progress"] = 95.0
        on_update(job)

        ext = settings.get("container", "mp4")
        final_name = f"enhanced_{job_id}_{Path(src).stem}.{ext}"
        final_path = DIRS["outputs"] / final_name

        audio_mode = settings.get("audio_mode", "original")
        has_orig_audio = info["has_audio"] and (audio_mode == "original")

        export_cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(assembled_video)]
        if has_orig_audio:
            export_cmd += ["-i", src, "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-b:a", f"{settings.get('audio_bitrate_k', 256)}k", "-af", "apad", "-shortest"]
        else:
            export_cmd += ["-map", "0:v:0", "-an"]

        export_cmd += [
            "-c:v", settings.get("vcodec", "libx264"),
            "-crf", str(settings.get("crf", 14)),
            "-preset", settings.get("x_preset", "slow"),
            "-pix_fmt", settings.get("pix_fmt", "yuv420p")
        ]
        if settings.get("faststart", True):
            export_cmd += ["-movflags", "+faststart"]
        export_cmd += [str(final_path)]

        subprocess.run(export_cmd, check=True)

        # 8. Generate Before/After Comparison Preview
        job["stage"] = "Generating before/after comparisons..."
        preview_img = work_dir / "preview_split.jpg"
        subprocess.run([
            "ffmpeg", "-y", "-v", "error",
            "-ss", "0.5", "-i", src,
            "-ss", "0.5", "-i", str(final_path),
            "-filter_complex", f"[0:v]scale={tw}:{th}:flags=lanczos[v0];[v0][1:v]hstack[v]",
            "-map", "[v]", "-frames:v", "1", "-q:v", "3", str(preview_img)
        ], check=False)

        preview_url = None
        if preview_img.exists():
            pub_preview = DIRS["outputs"] / f"preview_{job_id}.jpg"
            shutil.copy2(preview_img, pub_preview)
            preview_url = f"/outputs/{pub_preview.name}"

        # 9. Clean up scratch if not requested to keep
        if not settings.get("keep_chunks", False):
            shutil.rmtree(work_dir, ignore_errors=True)

        # Finish
        t_total = time.time() - t_start
        job["status"] = "completed"
        job["progress"] = 100.0
        job["stage"] = "Enhancement complete!"
        job["eta_text"] = f"Finished in {int(t_total)}s"
        job["output_video_path"] = str(final_path)
        job["output_video_url"] = f"/outputs/{final_name}"
        job["preview_url"] = preview_url
        job["elapsed_seconds"] = round(t_total, 1)
        on_update(job)

        # Release VRAM if configured to free on idle
        cfg = load_config()
        if cfg.get("release_vram_idle_s", 0) > 0:
            self._idle_free_task = asyncio.create_task(self._delayed_vram_free(cfg["release_vram_idle_s"], self._idle_generation))

    async def _delayed_vram_free(self, delay: int, generation: int):
        await asyncio.sleep(delay)
        async with self._execution_lock:
            if generation != self._idle_generation:
                return
            try:
                queue = await self.comfy.queue()
                if not isinstance(queue, dict) or any(not isinstance(queue.get(k), list) for k in ("queue_running", "queue_pending")):
                    return
                if queue["queue_running"] or queue["queue_pending"]:
                    return
                await self.comfy.free(unload=True)
            except (aiohttp.ClientError, asyncio.TimeoutError):
                return  # Retry on a later idle period; never guess that the worker is idle.
