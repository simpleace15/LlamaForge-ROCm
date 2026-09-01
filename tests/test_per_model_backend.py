"""Per-model AMD backend selection (device= in models.ini sections).

llama.cpp's router merges its OWN CLI args into every child process
(server-models.cpp:551 preset.merge(base_preset)) and LLAMA_ARG_DEVICE is not
stripped by unset_reserved_args - so a router-level --device silently
overwrites every per-section device (verified live 2026-08-31). The rule that
falls out: when ANY model section carries its own `device=` key, the router
must launch with NO --device flag at all, so section values survive.
"""
import conftest_paths  # noqa: F401
import unittest
import hardware


class TestPerModelBackend(unittest.TestCase):
    # ---- section parsing -------------------------------------------------
    def test_sections_with_device_detected(self):
        ini = {"*": {"ctx-size": "8192"},
               "big-mtp": {"model": "/m/a.gguf", "device": "Vulkan0,Vulkan1"},
               "aux": {"model": "/m/b.gguf", "device": "ROCm0"}}
        self.assertTrue(hardware.ini_defines_per_model_device(ini))

    def test_sections_without_device(self):
        ini = {"*": {"ctx-size": "8192"},
               "plain": {"model": "/m/a.gguf", "ctx-size": "4096"}}
        self.assertFalse(hardware.ini_defines_per_model_device(ini))

    def test_empty_ini(self):
        self.assertFalse(hardware.ini_defines_per_model_device({}))

    def test_global_star_device_does_not_count(self):
        # [*] is the global cascade layer, not a per-model override.
        ini = {"*": {"device": "ROCm0,ROCm1"}, "m": {"model": "/m/a.gguf"}}
        self.assertFalse(hardware.ini_defines_per_model_device(ini))

    def test_comments_and_blank_values_ignored(self):
        # read_sections() hands back whatever text is in the file; a commented
        # or empty device key is not a per-model selection.
        ini = {"m": {"device": "", "# device": "Vulkan0"}}
        self.assertFalse(hardware.ini_defines_per_model_device(ini))

    def test_star_section_device_does_not_count(self):
        ini = {"*": {"n-gpu-layers": "99"},
               "m": {"model": "/m/a.gguf"}}
        self.assertFalse(hardware.ini_defines_per_model_device(ini))

    # ---- router device suppression ---------------------------------------
    def test_router_device_empty_when_per_model(self):
        ini = {"*": {}, "m": {"device": "Vulkan0"}}
        self.assertEqual(
            hardware.router_device_for("rocm", 3, per_model=True), "")

    def test_router_device_normal_when_no_per_model(self):
        self.assertEqual(
            hardware.router_device_for("rocm", 3, per_model=False),
            "ROCm0,ROCm1,ROCm2")
        self.assertEqual(
            hardware.router_device_for("vulkan", 3, per_model=False),
            "Vulkan0,Vulkan1,Vulkan2")

    def test_router_device_zero_gpus_empty_even_per_model(self):
        # No AMD GPUs detected -> nothing to pass either way.
        self.assertEqual(
            hardware.router_device_for("rocm", 0, per_model=True), "")


if __name__ == "__main__":
    unittest.main()