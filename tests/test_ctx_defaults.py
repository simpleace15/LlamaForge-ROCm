"""apply_ctx_defaults() must backfill without clobbering.

It runs on every panel startup (server.py main()), so anything it rewrites it
rewrites behind the user's back. Its job is to give models a sane ctx-size when
they have none - not to overrule one the user chose deliberately.

The regression this pins: a model whose GGUF trained length exceeds the global
default had its explicit per-model ctx-size DELETED so it would "inherit the
global". Trained length is not a VRAM budget - a 27B Q6_K that trains to 262144
still OOMs at 150000 on a 32 GB box - so the deletion silently reimposed a
config that could not load.
"""
import conftest_paths  # noqa: F401
import json, os, tempfile, unittest
from unittest import mock

import config


class ApplyCtxDefaultsTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "models.ini")

    def _write(self, text):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(text)

    def _read(self):
        return config.read_sections(self.path)

    def _run(self, default_ctx):
        """default_ctx: what gguf.default_ctx returns for every model here."""
        with mock.patch.object(config.gguf, "default_ctx", return_value=default_ctx):
            return config.apply_ctx_defaults(self.path)

    def test_sets_the_global_baseline(self):
        self._write("[a]\nmodel = /m/a.gguf\n")
        self._run(0)
        self.assertEqual(self._read()["*"]["ctx-size"], config.CTX_GLOBAL_DEFAULT)

    def test_keeps_an_explicit_ctx_size_on_a_model_that_could_reach_the_global(self):
        """The regression. 65536 is there because 150000 does not fit in VRAM."""
        self._write("[*]\nctx-size = 150000\n\n[a]\nctx-size = 65536\nmodel = /m/a.gguf\n")
        self._run(0)
        self.assertEqual(self._read()["a"].get("ctx-size"), "65536")

    def test_still_backfills_a_model_that_cannot_reach_the_global(self):
        self._write("[*]\nctx-size = 150000\n\n[a]\nmodel = /m/a.gguf\n")
        self._run(40000)
        self.assertEqual(self._read()["a"].get("ctx-size"), "40000")

    def test_clamps_a_value_that_over_extends_the_trained_length(self):
        """Safety kept: never ask for more context than the model was trained on."""
        self._write("[*]\nctx-size = 150000\n\n[a]\nctx-size = 99999\nmodel = /m/a.gguf\n")
        self._run(40000)
        self.assertEqual(self._read()["a"].get("ctx-size"), "40000")

    def test_leaves_a_smaller_deliberate_value_below_the_cap(self):
        self._write("[*]\nctx-size = 150000\n\n[a]\nctx-size = 8192\nmodel = /m/a.gguf\n")
        self._run(40000)
        self.assertEqual(self._read()["a"].get("ctx-size"), "8192")

    def test_leaves_models_with_an_unreadable_trained_length_alone(self):
        self._write("[*]\nctx-size = 150000\n\n[a]\nctx-size = 4096\nmodel = /m/a.gguf\n")
        self._run(None)
        self.assertEqual(self._read()["a"].get("ctx-size"), "4096")

    def test_is_idempotent(self):
        self._write("[*]\nctx-size = 150000\n\n[a]\nctx-size = 65536\nmodel = /m/a.gguf\n")
        self._run(0)
        second = self._run(0)
        self.assertEqual(second["changed"], [],
                         "a second pass rewrote sections it had already settled")


class ConfiguredGlobalCtxTest(unittest.TestCase):
    """config.json `ctx_size` drives the [*] global, so lowering the default
    (to fit more resident models in VRAM) survives a restart instead of being
    rewritten back to 150000."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "models.ini")
        self._saved_config = config.CONFIG
        config.CONFIG = os.path.join(self.dir, "config.json")
        with open(config.CONFIG, "w", encoding="utf-8") as f:
            json.dump({"ctx_size": 32768}, f)

    def tearDown(self):
        config.CONFIG = self._saved_config

    def _write(self, text):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(text)

    def test_global_ctx_size_reads_config(self):
        self.assertEqual(config.global_ctx_size(), 32768)

    def test_apply_uses_configured_global_not_150000(self):
        self._write("[a]\nmodel = /m/a.gguf\n")
        with mock.patch.object(config.gguf, "default_ctx", return_value=0):
            config.apply_ctx_defaults(self.path)
        self.assertEqual(config.read_sections(self.path)["*"]["ctx-size"], "32768")

    def test_missing_ctx_size_falls_back_to_150000(self):
        with open(config.CONFIG, "w", encoding="utf-8") as f:
            json.dump({}, f)
        self.assertEqual(config.global_ctx_size(), 150000)
