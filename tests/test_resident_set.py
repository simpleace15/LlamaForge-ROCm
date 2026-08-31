"""Resident-set VRAM planner: can models_max x selected models fit in VRAM?

The router keeps up to `models_max` models resident (LRU eviction). Each
resident child holds weights + KV cache. On the 90 GB (3x V620) target the
measured ceiling is ~75 GB co-resident (122B-MTP IQ4_XS 62 GB + 4B 3 GB +
KV). This module sums predicted weight footprints of all registered models
and returns a per-model warning when they can't co-reside.
"""
import conftest_paths  # noqa: F401
import unittest
from unittest import mock

import routes
import vram_predict


def _giB(x):
    return x * 1024**3


class ResidentSetTest(unittest.TestCase):
    """routes.resident_set_warning() — the models.ini feasibility check."""

    def _patch(self, sections, gpu_gib=90, predict_map=None):
        """Patch config + GPU inventory + per-model size prediction."""
        self.sections = sections
        self.pred = predict_map or {}
        p1 = mock.patch.object(routes.config, "read_sections",
                               return_value=sections)
        p2 = mock.patch.object(routes.hardware, "detect_amd_gpus",
                               return_value=[{"vram_mib": gpu_gib * 1024}] * 3)
        p3 = mock.patch.object(
            routes.vram_predict, "predict_local",
            side_effect=lambda path, **kw: {
                "model_gb": self.pred.get(path, 0.0), "vram_gb": self.pred.get(path, 0.0),
                "regime": "gpu-resident"})
        for p in (p1, p2, p3):
            p.start(); self.addCleanup(p.stop)

    def test_no_warning_when_set_fits(self):
        ini = {"*": {},
               "small-a": {"model": "/m/a.gguf"},
               "small-b": {"model": "/m/b.gguf"}}
        with mock.patch.object(routes.config, "read_sections", return_value=ini), \
             mock.patch.object(routes.hardware, "detect_amd_gpus",
                               return_value=[{"vram_mib": 90 * 1024}] * 3), \
             mock.patch.object(routes.vram_predict, "predict_local",
                               return_value={"model_gb": 3.0}):
            out = routes.resident_set_warning()
        self.assertIsNone(out)

    def test_warns_when_models_exceed_pool(self):
        # Two 62 GB models with models_max=2: 124 GB > ~76 GB usable pool.
        import os, tempfile
        tmp = tempfile.mkdtemp(prefix="lf-resident-")
        big_a = os.path.join(tmp, "big-a.gguf")
        big_b = os.path.join(tmp, "big-b.gguf")
        for path in (big_a, big_b):
            with open(path, "wb") as f:
                f.truncate(int(62.0 * 1024**3))
        self.addCleanup(lambda: (os.remove(big_a), os.remove(big_b)))
        ini = {"*": {},
               "big-122b-a": {"model": big_a},
               "big-122b-b": {"model": big_b}}   # two distinct 62 GB models
        with mock.patch.object(routes.config, "read_sections", return_value=ini), \
             mock.patch.object(routes.hardware, "detect_amd_gpus",
                               return_value=[{"vram_mib": 30720}] * 3):
            out = routes.resident_set_warning(models_max=2)
        self.assertIsNotNone(out)
        self.assertIn("will not co-reside", out)

    def test_no_warning_without_gguf_paths(self):
        ini = {"*": {}, "m1": {"model": ""}}
        with mock.patch.object(routes.config, "read_sections", return_value=ini):
            self.assertIsNone(routes.resident_set_warning())


if __name__ == "__main__":
    unittest.main()