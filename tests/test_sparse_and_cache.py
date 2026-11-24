import sys
import unittest
from pathlib import Path
from unittest.mock import patch

STUDIO_DIR = Path(__file__).resolve().parents[1] / "studio"
if str(STUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIO_DIR))

import comfy as comfy_mod
import common
import graph
import settings


class SparseAndCacheTests(unittest.TestCase):
    def test_defaults_enable_ck_int8_and_ram_cache(self):
        self.assertEqual(common.DEFAULT_CONFIG["attention"], "ck-int8")
        self.assertEqual(common.DEFAULT_CONFIG["cache_mode"], "ram")
        self.assertTrue(settings.DEFAULTS["sparse_attention"])
        self.assertAlmostEqual(settings.DEFAULTS["sparse_tau"], 1.3)
        self.assertAlmostEqual(settings.DEFAULTS["sparse_start_percent"], 0.2)

    def test_normalize_clamps_sparse_settings(self):
        clean, _ = settings.normalize({"sparse_attention": True, "sparse_tau": 9.0, "sparse_start_percent": -1.0})
        self.assertTrue(clean["sparse_attention"])
        self.assertEqual(clean["sparse_tau"], 2.5)
        self.assertEqual(clean["sparse_start_percent"], 0.0)
        self.assertIn("sparse_attention", settings.graph_hash_keys())

    def test_refine_and_restore_wire_block_sparse_attention(self):
        with patch.object(graph, "ram_limits", return_value=(41.8, 30.0)):
            for mode in ("refine", "restore"):
                s, _ = settings.normalize({"mode": mode, "sparse_attention": True})
                g = graph.build_workflow(s, "in.mp4", "out", 1920, 1088, 1280, 704, sparse_available=True)
                self.assertIn("5004:5607", g)
                node = g["5004:5607"]
                self.assertEqual(node["class_type"], "BlockSparseAttention")
                self.assertEqual(node["inputs"]["model"], ["5004:5606", 0])
                self.assertEqual(node["inputs"]["selection"], "sol-attn")
                self.assertEqual(node["inputs"]["selection.tau"], 1.3)
                self.assertEqual(node["inputs"]["start_percent"], 0.2)
                self.assertEqual(node["inputs"]["min_tokens"], 4096)
                self.assertEqual(node["inputs"]["extra_tokens"], 256)
                self.assertEqual(node["inputs"]["sink_conditioning"], "off")
                self.assertEqual(g["5516:9100"]["inputs"]["model"], ["5004:5607", 0])

    def test_pixel_mode_wires_block_sparse_attention_into_guider(self):
        with patch.object(graph, "ram_limits", return_value=(41.8, 30.0)):
            s, _ = settings.normalize({"mode": "pixel", "sparse_attention": True, "sparse_tau": 1.5})
            g = graph.build_workflow(s, "in.mp4", "out", 1920, 1088, 960, 544, first_frame_image_name="f0.png", sparse_available=True)
            self.assertIn("5004:5607", g)
            self.assertEqual(g["5004:5607"]["inputs"]["selection.tau"], 1.5)
            self.assertEqual(g["5516:4828"]["inputs"]["model"], ["5004:5607", 0])

    def test_sparse_attention_omitted_when_disabled_or_unavailable(self):
        with patch.object(graph, "ram_limits", return_value=(41.8, 30.0)):
            s_off, _ = settings.normalize({"mode": "refine", "sparse_attention": False})
            g_off = graph.build_workflow(s_off, "in.mp4", "out", 1920, 1088, 1280, 704, sparse_available=True)
            self.assertNotIn("5004:5607", g_off)
            self.assertEqual(g_off["5516:9100"]["inputs"]["model"], ["5004:5606", 0])

            s_on, _ = settings.normalize({"mode": "refine", "sparse_attention": True})
            g_unavail = graph.build_workflow(s_on, "in.mp4", "out", 1920, 1088, 1280, 704, sparse_available=False)
            self.assertNotIn("5004:5607", g_unavail)
            self.assertEqual(g_unavail["5516:9100"]["inputs"]["model"], ["5004:5606", 0])

    def test_build_cmd_uses_ck_attention_and_bounded_cache_ram_on_low_ram(self):
        proc = object.__new__(comfy_mod.ComfyProcess)
        proc.capabilities = lambda: [
            "--reserve-vram", "--fp16-intermediates", "--use-ck-attention",
            "--use-pytorch-cross-attention", "--cache-ram", "--cache-none", "--disable-auto-launch",
        ]
        cfg = {**common.DEFAULT_CONFIG, "attention": "ck-int8", "cache_mode": "ram", "comfy_extra_args": []}
        with patch.object(comfy_mod, "load_config", return_value=cfg), \
             patch.object(comfy_mod, "ram_limits", return_value=(41.8, 30.0)):
            cmd = proc.build_cmd()
        self.assertIn("--use-ck-attention", cmd)
        self.assertNotIn("--use-pytorch-cross-attention", cmd)
        idx = cmd.index("--cache-ram")
        self.assertEqual(cmd[idx:idx + 3], ["--cache-ram", "6", "12"])

        with patch.object(comfy_mod, "load_config", return_value=cfg), \
             patch.object(comfy_mod, "ram_limits", return_value=(96.0, 80.0)):
            cmd_hi = proc.build_cmd()
        idx_hi = cmd_hi.index("--cache-ram")
        self.assertNotEqual(cmd_hi[idx_hi:idx_hi + 3], ["--cache-ram", "6", "12"])


if __name__ == "__main__":
    unittest.main()
