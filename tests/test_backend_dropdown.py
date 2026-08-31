"""POST /api/model/backend — per-model AMD backend selector.

Writes device= (+ the backend's benchmark default flags) into the model's
models.ini section, then reloads the router. device="" means "auto": clear
both device and backend-default flags so llama.cpp auto-selects. Saving also
unloads a loaded model (device is read at load time by llama.cpp).
"""
import conftest_paths  # noqa: F401
import unittest
from unittest import mock

import config
import routes
from routes import Req


def _start_router(ini_path, models):
    """Spin the smoke-test router: per-model argv from /v1/models on 18090."""
    import subprocess, os, time, urllib.request, json
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ("/tmp/llamacpp/build-vk/bin:"
                              + env["HOME"] + "/.local/vulkan-runtime/usr/lib/x86_64-linux-gnu")
    proc = subprocess.Popen(
        ["/tmp/llamacpp/build-vk/bin/llama-server", "--models-preset", ini_path,
         "--models-max", "4", "--offline", "--host", "127.0.0.1", "--port", "18090",
         "--metrics"], env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2.5)
    return proc


class _FakeReq:
    def __init__(self, body):
        self.body = body


class BackendRouteTest(unittest.TestCase):
    def setUp(self):
        import tempfile, os
        self.tmp = tempfile.mkdtemp(prefix="lf-backend-test-")
        self.ini = os.path.join(self.tmp, "models.ini")
        with open(self.ini, "w") as f:
            f.write("version = 1\n\n[*]\nctx-size = 8192\nload-on-startup = false\n\n"
                    "[m1]\nmodel = /tmp/m1.gguf\n")
        self.base = {"router_port": 8080, "router_host": "127.0.0.1",
                     "router_api_key": "", "models_ini": self.ini,
                     "server_bin": "/bin/llama-server", "active_engine": "llamacpp"}
        self.stop_router = None
        # Dev/CI boxes have no AMD GPUs; pin the device count the presets use.
        self.gpus = mock.patch.object(
            routes.hardware, "detect_amd_gpus",
            return_value=[{"index": i, "name": "AMD", "vram_mib": 30720,
                           "gfx_arch": "gfx1030"} for i in range(3)]).start()
        self.addCleanup(mock.patch.stopall)

    def test_valid_backends_accepted(self):
        with mock.patch.object(config, "load", return_value=self.base), \
             mock.patch.object(config, "ini_path", return_value=self.ini):
            for backend in ("auto", "vulkan", "rocm"):
                res = routes.post_model_backend(
                    _FakeReq({"model": "m1", "backend": backend}))
                self.assertEqual(res[0], 200)

    def test_unknown_backend_rejected(self):
        with self.assertRaises(routes.ApiError) as cm:
            routes.post_model_backend(_FakeReq({"model": "m1", "backend": "cuda"}))
        self.assertEqual(cm.exception.status, 400)

    def test_vulkan_writes_benchmark_defaults(self):
        with mock.patch.object(config, "load", return_value=self.base), \
             mock.patch.object(config, "ini_path", return_value=self.ini):
            routes.post_model_backend(_FakeReq({"model": "m1", "backend": "vulkan"}))
        sect = config.read_sections(self.ini).get("m1", {})
        self.assertEqual(sect.get("device"), "Vulkan0,Vulkan1,Vulkan2")
        self.assertEqual(sect.get("split-mode"), "layer")
        self.assertEqual(sect.get("n-gpu-layers"), "99")
        self.assertEqual(sect.get("cache-type-k"), "q8_0")
        self.assertEqual(sect.get("cache-type-v"), "q8_0")
        self.assertEqual(sect.get("jinja"), "true")
        # Vulkan MoE guardrail: ngram speculation measured -11% -> never set
        self.assertNotIn("spec-type", sect)

    def test_rocm_writes_ub_batch_defaults(self):
        with mock.patch.object(config, "load", return_value=self.base), \
             mock.patch.object(config, "ini_path", return_value=self.ini):
            routes.post_model_backend(_FakeReq({"model": "m1", "backend": "rocm"}))
        sect = config.read_sections(self.ini).get("m1", {})
        self.assertEqual(sect.get("device"), "HIP0,HIP1,HIP2")
        self.assertEqual(sect.get("ubatch-size"), "1024")
        self.assertEqual(sect.get("batch-size"), "4096")

    def test_auto_clears_device_and_backend_flags(self):
        self.base["models_ini"] = self.ini
        with open(self.ini, "a") as f:
            f.write("\n[m1]\nmodel = /tmp/m1.gguf\ndevice = Vulkan0\nsplit-mode = layer\n")
        with mock.patch.object(config, "load", return_value=self.base), \
             mock.patch.object(config, "ini_path", return_value=self.ini):
            routes.post_model_backend(_FakeReq({"model": "m1", "backend": "auto"}))
        sect = config.read_sections(self.ini).get("m1", {})
        self.assertNotIn("device", sect)
        self.assertNotIn("split-mode", sect)


if __name__ == "__main__":
    unittest.main()