import conftest_paths  # noqa: F401
import unittest
import hardware, vram_predict as vp
from vramwise import constants as C


class TestAmdGfxName(unittest.TestCase):
    def test_gfx1030(self):
        self.assertEqual(hardware._amd_gfx_name(103000), "gfx1030")

    def test_gfx90a(self):
        self.assertEqual(hardware._amd_gfx_name(90010), "gfx90a")

    def test_gfx942(self):
        self.assertEqual(hardware._amd_gfx_name(90402), "gfx942")

    def test_gfx1100(self):
        self.assertEqual(hardware._amd_gfx_name(110000), "gfx1100")

    def test_unparseable(self):
        self.assertEqual(hardware._amd_gfx_name(""), "")
        self.assertEqual(hardware._amd_gfx_name(None), "")


class TestAmdTargets(unittest.TestCase):
    def test_dedup_sorted(self):
        gpus = [{"gfx_arch": "gfx1100"}, {"gfx_arch": "gfx1030"}, {"gfx_arch": "gfx1100"}]
        self.assertEqual(hardware._amd_targets_for(gpus), "gfx1030;gfx1100")

    def test_empty_when_unknown(self):
        self.assertEqual(hardware._amd_targets_for([{"gfx_arch": ""}]), "")


class TestRecommendRocm(unittest.TestCase):
    def test_amd_gpus_set_hip_flags(self):
        amd = [{"index": 0, "name": "AMD Instinct MI300X", "vram_mib": 196608,
                "gfx_arch": "gfx942"}]
        r = hardware.recommend(gpus=[], cpu={"avx512_hint": False}, amd_gpus=amd)
        self.assertEqual(r["cmake_flags"]["GGML_HIP"], "ON")
        self.assertIn("AMDGPU_TARGETS", r["cmake_flags"])
        self.assertEqual(r["runtime"]["n-gpu-layers"], "99")
        self.assertEqual(r["runtime"]["flash-attn"], "on")

    def test_amd_targets_override_wins(self):
        amd = [{"index": 0, "name": "AMD", "vram_mib": 16384, "gfx_arch": "gfx1100"}]
        r = hardware.recommend(gpus=[], cpu={"avx512_hint": False}, amd_gpus=amd,
                               amd_targets="gfx1030;gfx1100")
        self.assertEqual(r["cmake_flags"]["AMDGPU_TARGETS"], "gfx1030;gfx1100")

    def test_amd_detected_archs_used_when_no_override(self):
        amd = [{"index": 0, "name": "AMD", "vram_mib": 16384, "gfx_arch": "gfx1100"}]
        r = hardware.recommend(gpus=[], cpu={"avx512_hint": False}, amd_gpus=amd)
        self.assertEqual(r["cmake_flags"]["AMDGPU_TARGETS"], "gfx1100")

    def test_amd_falls_back_to_default_targets(self):
        amd = [{"index": 0, "name": "AMD", "vram_mib": 16384, "gfx_arch": ""}]
        r = hardware.recommend(gpus=[], cpu={"avx512_hint": False}, amd_gpus=amd)
        self.assertEqual(r["cmake_flags"]["AMDGPU_TARGETS"], hardware.AMDGPU_TARGETS_DEFAULT)

    def test_cuda_still_works(self):
        nv = [{"index": 0, "name": "RTX 4090", "vram_mib": 24576, "compute_cap": "8.9"}]
        r = hardware.recommend(gpus=nv, cpu={"avx512_hint": False}, amd_gpus=[])
        self.assertEqual(r["cmake_flags"]["GGML_CUDA"], "ON")
        self.assertEqual(r["cmake_flags"]["CMAKE_CUDA_ARCHITECTURES"], "89")
        self.assertNotIn("GGML_HIP", r["cmake_flags"])

    def test_cpu_only_when_no_gpus(self):
        r = hardware.recommend(gpus=[], cpu={"avx512_hint": False}, amd_gpus=[])
        self.assertNotIn("GGML_CUDA", r["cmake_flags"])
        self.assertNotIn("GGML_HIP", r["cmake_flags"])
        self.assertEqual(r["runtime"]["n-gpu-layers"], "0")


class TestAmdVramBandwidth(unittest.TestCase):
    def test_mi300x(self):
        self.assertEqual(vp._preset_vram_bw("AMD Instinct MI300X"), 5300)

    def test_rx7900xtx(self):
        self.assertEqual(vp._preset_vram_bw("AMD Radeon RX 7900 XTX"), 960)

    def test_unknown_amd_falls_back(self):
        self.assertEqual(vp._preset_vram_bw("AMD Radeon RX 9999"), C.DEFAULT_VRAM_BW)

    def test_nvidia_still_resolves(self):
        self.assertEqual(vp._preset_vram_bw("NVIDIA GeForce RTX 4090"), 1008)


if __name__ == "__main__":
    unittest.main()
