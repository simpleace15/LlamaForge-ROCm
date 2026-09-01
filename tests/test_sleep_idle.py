"""Sleep-on-idle default for small models (VRAM relief under contention).

Upstream llama.cpp --sleep-idle-seconds (PR #18228): after N idle seconds a
child in router mode sleeps — model + KV memory released, auto-reloads on
the next request. post_model_backend fills 900 s for <= 16 GiB models
when the section carries no hand-set value.
"""
import conftest_paths  # noqa: F401
import unittest
from unittest import mock

import config
import routes


class _FakeReq:
    def __init__(self, body):
        self.body = body


class SleepIdleDefaultTest(unittest.TestCase):
    def setUp(self):
        import os, tempfile
        self.tmp = tempfile.mkdtemp(prefix="lf-sleep-")
        self.ini = os.path.join(self.tmp, "models.ini")
        self.small = os.path.join(self.tmp, "small.gguf")   # 3 GiB -> voice-sized
        self.big = os.path.join(tmp_big := os.path.join(self.tmp, "big"), "big.gguf")
        os.makedirs(tmp_big, exist_ok=True)
        with open(self.small, "wb") as f:
            f.truncate(3 * 1024**3)
        with open(self.big, "wb") as f:
            f.truncate(62 * 1024**3)
        with open(self.ini, "w") as f:
            f.write("version = 1\n\n[*]\n\n"
                    f"[voice-4b]\nmodel = {self.small}\n\n"
                    f"[big-122b]\nmodel = {self.big}\n")
        self.base = {"router_port": 8080, "router_host": "127.0.0.1",
                     "router_api_key": "", "models_ini": self.ini,
                     "server_bin": "/bin/llama-server", "active_engine": "llamacpp"}
        self.gpus = mock.patch.object(
            routes.hardware, "detect_amd_gpus",
            return_value=[{"index": i, "name": "AMD", "vram_mib": 30720,
                           "gfx_arch": "gfx1030"} for i in range(3)]).start()
        self.addCleanup(mock.patch.stopall)

    def test_small_model_gets_sleep_idle_default(self):
        with mock.patch.object(config, "load", return_value=self.base), \
             mock.patch.object(config, "ini_path", return_value=self.ini):
            routes.post_model_backend(_FakeReq({"model": "voice-4b",
                                                "backend": "vulkan"}))
        sect = config.read_sections(self.ini)["voice-4b"]
        self.assertEqual(sect.get("sleep-idle-seconds"), "900")

    def test_big_model_gets_no_sleep_default(self):
        with mock.patch.object(config, "load", return_value=self.base), \
             mock.patch.object(config, "ini_path", return_value=self.ini):
            routes.post_model_backend(_FakeReq({"model": "big-122b",
                                                "backend": "rocm"}))
        sect = config.read_sections(self.ini)["big-122b"]
        self.assertNotIn("sleep-idle-seconds", sect)

    def test_existing_sleep_value_never_overridden(self):
        # hand-set value in the model SECTION (not [*]) must be preserved
        with open(self.ini, "w") as f:
            f.write("version = 1\n\n[*]\n\n"
                    f"[voice-4b]\nmodel = {self.small}\nsleep-idle-seconds = 60\n\n"
                    f"[big-122b]\nmodel = {self.big}\n")
        with mock.patch.object(config, "load", return_value=self.base), \
             mock.patch.object(config, "ini_path", return_value=self.ini):
            routes.post_model_backend(_FakeReq({"model": "voice-4b",
                                                "backend": "vulkan"}))
        sect = config.read_sections(self.ini)["voice-4b"]
        self.assertEqual(sect.get("sleep-idle-seconds"), "60")


if __name__ == "__main__":
    unittest.main()