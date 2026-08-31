"""GLM-family min-p default: min-p 0.01 (llama.cpp's 0.05 harms GLM)."""
import conftest_paths  # noqa: F401
import unittest
from unittest import mock

import config
import glm_defaults


class GlmDefaultsTest(unittest.TestCase):
    def test_writes_min_p_only_for_glm(self):
        seen = {}
        ini = {"glm-4.7-flash": {"model": "/models/glm.gguf"},
               "qwen3-30b": {"model": "/models/qwen.gguf"}}
        out = glm_defaults.apply_glm_defaults(ini, set_keys=lambda s, u: seen.update({s: u}))
        self.assertEqual(out, ["glm-4.7-flash"])
        self.assertEqual(seen, {"glm-4.7-flash": {"min-p": "0.01"}})

    def test_existing_min_p_never_overridden(self):
        ini = {"glm-4.7-flash": {"model": "/m/g.gguf", "min-p": "0.05"}}
        out = glm_defaults.apply_glm_defaults(ini, set_keys=lambda s, u: None)
        self.assertEqual(out, [])

    def test_idempotent(self):
        """A section already carrying min-p 0.01 is not reported as changed."""
        changed = glm_defaults.apply_glm_defaults(
            {"glm-4-9b": {"model": "/m/g.gguf", "min-p": "0.01"}},
            set_keys=lambda s, u: None)
        self.assertEqual(changed, [])

    def test_glm_detected_from_model_path(self):
        changed = glm_defaults.apply_glm_defaults(
            {"hybrid-name": {"model": "/models/GLM-Hybrid-Q4.gguf"}},
            set_keys=lambda s, u: None)
        self.assertEqual(changed, ["hybrid-name"])

    def test_star_section_ignored(self):
        changed = glm_defaults.apply_glm_defaults(
            {"*": {"min-p": ""}}, set_keys=lambda s, u: None)
        self.assertEqual(changed, [])

    def test_wrapper_extends_apply_ctx_defaults(self):
        """The monkeypatched config.apply_ctx_defaults runs the GLM pass too."""
        ini = {"glm-4.7-flash": {"model": "/m/g.gguf"}}
        with mock.patch.object(config, "read_sections", return_value=ini), \
             mock.patch.object(config, "apply_ctx_defaults",
                               return_value={"changed": ["*"]}) as orig:
            import glm_defaults as gd
            # re-wrap from the pristine original to avoid double-patching
            res = gd.apply_ctx_defaults_with_glm(None)
        self.assertIn("glm-4.7-flash", res["changed"])


if __name__ == "__main__":
    unittest.main()