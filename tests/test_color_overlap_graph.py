import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

STUDIO_DIR = Path(__file__).resolve().parents[1] / "studio"
if str(STUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIO_DIR))

import colorfix
import graph
import settings
import video


class GraphUpgradesTests(unittest.TestCase):
    def test_refine_prunes_dead_audio_vae_when_original_or_none(self):
        with patch.object(graph, "ram_limits", return_value=(41.8, 30.0)):
            for amode in ("original", "none"):
                s, _ = settings.normalize({"mode": "refine", "audio_mode": amode})
                g = graph.build_workflow(s, "in.mp4", "out", 1920, 1088, 1280, 704, sparse_available=True)
                self.assertNotIn("audio", g["5518:4849"]["inputs"])
                for dead_id in ("5004:5600", "9002:9007", "9002:9008", "5516:9102", "5518:4848", "5544", "9101"):
                    self.assertNotIn(dead_id, g, f"{dead_id} should be pruned for audio_mode={amode}")

    def test_refine_keeps_audio_vae_when_generated(self):
        with patch.object(graph, "ram_limits", return_value=(41.8, 30.0)):
            s, _ = settings.normalize({"mode": "refine", "audio_mode": "generated"})
            g = graph.build_workflow(s, "in.mp4", "out", 1920, 1088, 1280, 704, sparse_available=True)
            self.assertEqual(g["5518:4849"]["inputs"].get("audio"), ["5518:4848", 0])
            self.assertIn("5004:5600", g)
            self.assertIn("5518:4848", g)

    def test_custom_prompt_prunes_duplicate_clip_loader_and_previews(self):
        with patch.object(graph, "ram_limits", return_value=(41.8, 30.0)):
            s, _ = settings.normalize({"mode": "refine", "prompt_positive": "cinematic crisp detail", "audio_mode": "original"})
            self.assertFalse(s["cached_cond"])
            g = graph.build_workflow(s, "in.mp4", "out", 1920, 1088, 1280, 704, sparse_available=True)
            self.assertIn("5004:5605", g)
            self.assertNotIn("5004:5604", g)
            self.assertNotIn("5014:9004", g)
            self.assertNotIn("5004:5600", g)
            self.assertNotIn("5544", g)
            self.assertNotIn("9101", g)

    def test_pixel_mode_prunes_audio_decode_but_keeps_av_concat(self):
        with patch.object(graph, "ram_limits", return_value=(41.8, 30.0)):
            s, _ = settings.normalize({"mode": "pixel", "audio_mode": "original"})
            g = graph.build_workflow(s, "in.mp4", "out", 1920, 1088, 960, 544, first_frame_image_name="f0.png", sparse_available=True)
            self.assertNotIn("audio", g["5518:4849"]["inputs"])
            self.assertNotIn("5518:4848", g)
            self.assertIn("9002:4528", g)

    def test_tile_frames_synced_between_guide_and_sampler(self):
        with patch.object(graph, "ram_limits", return_value=(41.8, 30.0)):
            s, _ = settings.normalize({"mode": "refine", "tile_frames": 97, "overlap_frac": 0.1, "decode_temporal": 16})
            g = graph.build_workflow(s, "in.mp4", "out", 1920, 1088, 1280, 704, sparse_available=True)
            self.assertEqual(g["5516:9100"]["inputs"]["tile_frames"], 97)
            self.assertEqual(g["9002:5012"]["inputs"]["tile_frames"], 97)
            self.assertEqual(g["5516:9100"]["inputs"]["overlap_frac"], 0.25)
            self.assertLess(g["5518:5538"]["inputs"]["temporal_overlap"], g["5518:5538"]["inputs"]["temporal_size"])


class OverlapChunkingTests(unittest.TestCase):
    def test_overlap_chunks_preserve_exact_total_frames(self):
        chunks = video.smart_chunking(
            total_frames=240, fps=24.0, scene_cuts_s=[], budget_fmp=250.0,
            tw=1920, th=1088, refine=True, overlap_frames=8
        )
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(sum(c["frames"] for c in chunks), 240)
        self.assertEqual(chunks[0]["overlap_prev"], 0)
        for c in chunks[1:]:
            self.assertEqual(c["overlap_prev"], 8)
            self.assertEqual(c["extract_start_frame"], c["start_frame"] - 8)
            self.assertEqual(c["extract_frames"], c["frames"] + 8)
            self.assertEqual(c["render_frames"], video.render_length_8n1(c["extract_frames"]) + 8)

    def test_scene_cut_boundary_disables_overlap(self):
        # 200 frames at 24fps (~8.33s) with a scene cut right in the middle at frame 100
        # budget=270 => maxf ~106 frames => 2 chunks split cleanly on frame 100
        chunks = video.smart_chunking(
            total_frames=200, fps=24.0, scene_cuts_s=[100 / 24.0], budget_fmp=270.0,
            tw=1920, th=1088, refine=True, overlap_frames=8
        )
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[1]["start_frame"], 100)
        self.assertTrue(chunks[1]["is_cut_boundary"])
        self.assertEqual(chunks[1]["overlap_prev"], 0)
        self.assertEqual(chunks[1]["extract_frames"], chunks[1]["frames"])

    def test_zero_overlap_matches_legacy_hard_cuts(self):
        chunks = video.smart_chunking(
            total_frames=240, fps=24.0, scene_cuts_s=[], budget_fmp=250.0,
            tw=1920, th=1088, refine=True, overlap_frames=0
        )
        self.assertGreaterEqual(len(chunks), 2)
        for c in chunks:
            self.assertEqual(c["overlap_prev"], 0)
            self.assertEqual(c["extract_frames"], c["frames"])


class WaveletAndStitchTests(unittest.TestCase):
    def test_wavelet_fixes_spatially_varying_color_shift_and_preserves_detail(self):
        h, w = 256, 384
        src = np.zeros((h, w, 3), dtype=np.uint8)
        # Left half warm terracotta, right half cool slate blue
        src[:, :w // 2] = (175, 118, 92)
        src[:, w // 2:] = (82, 118, 165)

        # Simulated enhanced output: opposite local shifts on left vs right + crisp high-frequency grain/edges
        yy, xx = np.mgrid[0:h, 0:w]
        hf = (((xx + yy) % 2) * 2 - 1).astype(np.float32) * 14.0
        out = src.astype(np.float32).copy()
        out[:, :w // 2, 0] += 24.0  # Left gets too bright/red
        out[:, :w // 2, 2] -= 18.0
        out[:, w // 2:, 1] += 22.0  # Right gets green cast
        out[:, w // 2:, 0] -= 16.0
        out = np.clip(out + hf[..., None], 0, 255).astype(np.uint8)

        src_small = src[::2, ::2].copy()
        fixed = colorfix.wavelet_color_fix_frame(out, src_small, strength=1.0, max_shift=20.0)

        # Measure local Lab error on left and right interior regions (away from center step)
        src_L_l = colorfix.lab_stats(src[20:-20, 20:w // 2 - 30])[:3]
        out_L_l = colorfix.lab_stats(out[20:-20, 20:w // 2 - 30])[:3]
        fix_L_l = colorfix.lab_stats(fixed[20:-20, 20:w // 2 - 30])[:3]

        src_L_r = colorfix.lab_stats(src[20:-20, w // 2 + 30:-20])[:3]
        out_L_r = colorfix.lab_stats(out[20:-20, w // 2 + 30:-20])[:3]
        fix_L_r = colorfix.lab_stats(fixed[20:-20, w // 2 + 30:-20])[:3]

        err_before_l = float(np.linalg.norm(out_L_l - src_L_l))
        err_after_l = float(np.linalg.norm(fix_L_l - src_L_l))
        err_before_r = float(np.linalg.norm(out_L_r - src_L_r))
        err_after_r = float(np.linalg.norm(fix_L_r - src_L_r))

        self.assertGreater(err_before_l, 5.0)
        self.assertGreater(err_before_r, 5.0)
        self.assertLess(err_after_l, 1.5)
        self.assertLess(err_after_r, 1.5)

        # High-frequency energy (checkerboard micro-detail) must remain >90% intact
        out_gray = out.mean(axis=2)
        fix_gray = fixed.mean(axis=2)
        hf_out = float(np.abs(np.diff(out_gray, axis=1)).mean())
        hf_fix = float(np.abs(np.diff(fix_gray, axis=1)).mean())
        self.assertGreater(hf_fix / max(hf_out, 1e-6), 0.90)

    def test_seam_offset_applies_uniformly_even_inside_face_mask(self):
        rgb = np.full((64, 64, 3), 120, dtype=np.uint8)
        zero_mask = np.zeros((16, 16), dtype=np.float32)  # 0 = fully face-protected
        # Global offset is blocked by zero_mask, but seam_off must still equalize the face!
        fixed = colorfix.correct_frame(rgb, off=[10.0, 0.0, 0.0], gain=[1.0, 1.0, 1.0], wmap_small=zero_mask, seam_off=[5.0, 0.0, 0.0])
        base_L = colorfix.lab_stats(rgb)[0]
        fixed_L = colorfix.lab_stats(fixed)[0]
        self.assertAlmostEqual(fixed_L - base_L, 5.0, delta=0.8)

    def test_settings_defaults_and_clamping(self):
        self.assertEqual(settings.DEFAULTS["color_mode"], "wavelet")
        self.assertEqual(settings.DEFAULTS["chunk_overlap"], 8)
        self.assertFalse(settings.DEFAULTS["protect_faces"])
        self.assertEqual(settings.BUILTIN_PRESETS["old_footage"]["settings"]["color_mode"], "temporal")
        s_lo, _ = settings.normalize({"chunk_overlap": -5, "color_mode": "wavelet"})
        self.assertEqual(s_lo["chunk_overlap"], 0)
        s_hi, _ = settings.normalize({"chunk_overlap": 99})
        self.assertEqual(s_hi["chunk_overlap"], 16)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe required")
    def test_stitch_chunks_overlap_exact_frames_and_smooth_seam(self):
        w, h = 160, 96
        ov = 8
        with tempfile.TemporaryDirectory() as td:
            c0 = Path(td) / "c0.mp4"
            c1 = Path(td) / "c1.mp4"
            out = Path(td) / "stitched.mp4"

            def write_solid(path, n_frames, rgb_val):
                enc = subprocess.Popen([
                    "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{w}x{h}", "-r", "24", "-i", "-",
                    "-vf", "scale=in_color_matrix=bt709:out_color_matrix=bt709:in_range=full:out_range=tv",
                    "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                    str(path)
                ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                fr = np.full((h, w, 3), rgb_val, dtype=np.uint8).tobytes()
                for _ in range(n_frames):
                    enc.stdin.write(fr)
                enc.stdin.close()
                rc = enc.wait()
                enc.stderr.close()
                assert rc == 0

            # Chunk 0: 40 frames dark-neutral; Chunk 1: 48 frames (8 overlap + 40 new) with strong color/luma jump
            write_solid(c0, 40, (110, 110, 110))
            write_solid(c1, 48, (145, 132, 118))

            enc_cmd = [
                "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{w}x{h}", "-r", "24", "-i", "-",
                "-vf", "scale=in_color_matrix=bt709:out_color_matrix=bt709:in_range=full:out_range=tv",
                "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                str(out)
            ]
            specs = [
                {"path": str(c0), "extract_frames": 40, "overlap_prev": 0},
                {"path": str(c1), "extract_frames": 48, "overlap_prev": ov},
            ]
            res = colorfix.stitch_chunks_overlap(specs, enc_cmd, w, h, align_color=True, max_align_shift=15.0)
            self.assertEqual(res["frames"], 80)
            self.assertEqual(res["seams_blended"], 1)

            nb = int(subprocess.check_output([
                "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(out)
            ], text=True).strip())
            self.assertEqual(nb, 80)

            # Verify no sudden Lab jump at the boundary (frame 40)
            stats = colorfix.collect_stats(colorfix.lowres_cmd(["-i", str(out)], "", 80, 48), 80, 48)
            jumps = colorfix.seam_jumps(stats[:, 0], [40], cuts=[])
            self.assertEqual(len(jumps), 1)
            self.assertLess(abs(jumps[0]["jump"]), 2.5)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe required")
    def test_apply_wavelet_streaming_pipe_removes_chunk_jump(self):
        w, h = 160, 96
        with tempfile.TemporaryDirectory() as td:
            src_v = Path(td) / "src.mp4"
            out_v = Path(td) / "out.mp4"
            fix_v = Path(td) / "fix.mp4"

            def write_frames(path, frames):
                enc = subprocess.Popen([
                    "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{w}x{h}", "-r", "24", "-i", "-",
                    "-vf", "scale=in_color_matrix=bt709:out_color_matrix=bt709:in_range=full:out_range=tv",
                    "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                    str(path)
                ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                for fr in frames:
                    enc.stdin.write(fr.tobytes())
                enc.stdin.close()
                rc = enc.wait()
                enc.stderr.close()
                assert rc == 0

            # 24 source frames consistent warm neutral; output has chunk jump at frame 12 (+25 R, -15 B)
            src_fr = np.full((h, w, 3), (120, 120, 120), dtype=np.uint8)
            out_c0 = np.full((h, w, 3), (120, 120, 120), dtype=np.uint8)
            out_c1 = np.full((h, w, 3), (150, 125, 100), dtype=np.uint8)
            write_frames(src_v, [src_fr] * 24)
            write_frames(out_v, [out_c0] * 12 + [out_c1] * 12)

            dec_out = [
                "ffmpeg", "-v", "error", "-i", str(out_v), "-an",
                "-vf", "scale=in_color_matrix=bt709:out_color_matrix=bt709:in_range=tv:out_range=full,format=rgb24",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
            ]
            dec_src = colorfix.lowres_cmd(["-i", str(src_v)], "", 80, 48)
            enc_fix = [
                "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{w}x{h}", "-r", "24", "-i", "-",
                "-vf", "scale=in_color_matrix=bt709:out_color_matrix=bt709:in_range=full:out_range=tv",
                "-c:v", "libx264", "-crf", "8", "-preset", "medium", "-pix_fmt", "yuv420p",
                "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                str(fix_v)
            ]
            s_stats = colorfix.collect_stats(colorfix.lowres_cmd(["-i", str(src_v)], "", 80, 48), 80, 48)
            o_stats = colorfix.collect_stats(colorfix.lowres_cmd(["-i", str(out_v)], "", 80, 48), 80, 48)
            plan_ = colorfix.plan(s_stats, o_stats, cuts=[], bounds=[12], mode="wavelet", strength=0.95, max_shift=20.0)
            res = colorfix.apply_wavelet(
                dec_out, dec_src, enc_fix, w, h, 80, 48,
                strength=0.95, max_shift=20.0, seam_offsets=plan_["seam_offset"]
            )
            self.assertEqual(res["frames"], 24)
            self.assertEqual(res["corrected"], 24)

            f_stats = colorfix.collect_stats(colorfix.lowres_cmd(["-i", str(fix_v)], "", 80, 48), 80, 48)
            rep_before = colorfix.analyze(s_stats, o_stats, 24.0, cuts=[], bounds=[12])
            rep_after = colorfix.analyze(s_stats, f_stats, 24.0, cuts=[], bounds=[12])
            self.assertGreater(rep_before["drift"]["max_chroma"], 10.0)
            self.assertLess(rep_after["drift"]["max_chroma"], 2.5)
            self.assertEqual(rep_after["verdict"], "ok")


if __name__ == "__main__":
    unittest.main()
