"""models_max is user-configurable via /api/config with a 1..16 int check."""
import conftest_paths  # noqa: F401
import unittest
from unittest import mock

import config
import routes


class _Req:
    def __init__(self, body):
        self.body = body


class ModelsMaxConfigTest(unittest.TestCase):
    def test_accepts_valid_models_max(self):
        with mock.patch.object(routes.config, "update",
                               return_value=dict(routes.config.DEFAULTS)) as upd:
            status, out = routes.post_config(
                type("R", (), {"body": {"models_max": 6}})())
        self.assertTrue(out["ok"])
        self.assertIn("models_max", out["applied"])

    def test_rejects_zero_and_negatives(self):
        with self.assertRaises(routes.ApiError) as cm:
            routes.post_config(_Req({"models_max": 0}))
        self.assertEqual(cm.exception.status, 400)

    def test_rejects_bool(self):
        with self.assertRaises(routes.ApiError) as cm:
            routes.post_config(_FakeReq({"models_max": True}))
        self.assertEqual(cm.exception.status, 400)

    def test_rejects_non_int(self):
        with self.assertRaises(routes.ApiError) as cm:
            routes.post_config(_FakeReq({"models_max": "5"}))
        self.assertEqual(cm.exception.status, 400)


class _FakeReq:
    def __init__(self, body):
        self.body = body


if __name__ == "__main__":
    unittest.main()