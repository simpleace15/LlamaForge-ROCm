import conftest_paths  # noqa: F401
import json, os, tempfile, unittest, urllib.error, urllib.parse
import stats


class RouterCase(unittest.TestCase):
    def setUp(self):
        self._orig = stats.STATS_FILE
        fd, self.path = tempfile.mkstemp(suffix=".json"); os.close(fd); os.unlink(self.path)
        stats.STATS_FILE = self.path
        self.tr = stats.StatsTracker()
        self.tr._poll_vllm = lambda: None   # keep the test off the network

    def tearDown(self):
        stats.STATS_FILE = self._orig
        if os.path.exists(self.path):
            os.unlink(self.path)

    def _wire(self, prompt, gen, model="nomic", seen=None):
        """Emulate the llama.cpp router: bare /metrics 400s (needs a model
        name), /metrics?model= works, /models reports one loaded model."""
        def fake_get(path, timeout=4):
            if seen is not None:
                seen.append(path)
            if path == "/models":
                return json.dumps({"data": [
                    {"id": "default", "status": {"value": "unloaded"}},
                    {"id": model, "status": {"value": "loaded"}},
                ]})
            if path.startswith("/metrics?model="):
                return (f"llamacpp:prompt_tokens_total {prompt}\n"
                        f"llamacpp:tokens_predicted_total {gen}\n")
            if path == "/metrics":
                raise urllib.error.HTTPError(path, 400, "model name missing", {}, None)
            raise AssertionError("unexpected path " + path)
        self.tr._get = fake_get


class TestRouterMetricsScrape(RouterCase):
    def test_router_up_and_tokens_attributed(self):
        self._wire(prompt=10, gen=20)
        self.tr.poll_once()                       # baseline
        self.assertTrue(self.tr.live["router_up"])
        self._wire(prompt=15, gen=60)             # counters advanced
        self.tr.poll_once()
        m = self.tr.data["models"]["nomic"]
        self.assertEqual(m["prompt"], 5)
        self.assertEqual(m["generated"], 40)
        # Throughput is derived from the token deltas over POLL_SECS (5s), not
        # llama.cpp's decaying gauges: dp=5 -> 1.0 tok/s, dg=40 -> 8.0 tok/s.
        self.assertEqual(self.tr.live["prompt_per_sec"], 1.0)
        self.assertEqual(self.tr.live["gen_per_sec"], 8.0)

    def test_scrape_includes_model_param_never_bare(self):
        seen = []
        self._wire(prompt=1, gen=1, seen=seen)
        self.tr.poll_once()
        self.assertIn("/models", seen)
        self.assertTrue(any(p.startswith("/metrics?model=") for p in seen),
                        f"never scraped with ?model=; saw {seen}")
        self.assertNotIn("/metrics", seen)        # bare form must not be used

    def test_router_up_with_no_model_loaded(self):
        def fake_get(path, timeout=4):
            if path == "/models":
                return json.dumps({"data": [{"id": "default", "status": {"value": "unloaded"}}]})
            raise AssertionError("should not scrape metrics with nothing loaded")
        self.tr._get = fake_get
        self.tr.poll_once()
        self.assertTrue(self.tr.live["router_up"])   # up, just idle
        self.assertIsNone(self.tr.live["loaded_model"])

    def test_router_down_reports_offline(self):
        def fake_get(path, timeout=4):
            raise urllib.error.URLError("connection refused")
        self.tr._get = fake_get
        self.tr.poll_once()
        self.assertFalse(self.tr.live["router_up"])


class TestMultiModelScrape(RouterCase):
    """With --models-max > 1 several models can be resident at once; each
    model's tokens must be diffed against its own baseline and attributed to
    the right model."""

    def _wire_multi(self, models, seen=None):
        """models: dict id -> (prompt, gen). /models reports all as loaded;
        /metrics?model=<id> returns that model's counters."""
        def fake_get(path, timeout=4):
            if seen is not None:
                seen.append(path)
            if path == "/models":
                data = [{"id": "default", "status": {"value": "unloaded"}}]
                for mid in models:
                    data.append({"id": mid, "status": {"value": "loaded"}})
                return json.dumps({"data": data})
            if path.startswith("/metrics?model="):
                mid = urllib.parse.unquote(path.split("=", 1)[1])
                p, g = models[mid]
                return (f"llamacpp:prompt_tokens_total {p}\n"
                        f"llamacpp:tokens_predicted_total {g}\n")
            if path == "/metrics":
                raise urllib.error.HTTPError(path, 400, "model name missing", {}, None)
            raise AssertionError("unexpected path " + path)
        self.tr._get = fake_get

    def test_all_loaded_models_reported(self):
        self._wire_multi({"a": (0, 0), "b": (0, 0)})
        self.tr.poll_once()
        self.assertTrue(self.tr.live["router_up"])
        self.assertEqual(sorted(self.tr.live["loaded_models"]), ["a", "b"])
        self.assertEqual(self.tr.live["loaded_model"], "a")  # first, for back-compat

    def test_deltas_attributed_per_model(self):
        self._wire_multi({"a": (10, 20), "b": (100, 200)})
        self.tr.poll_once()                       # baseline both
        self._wire_multi({"a": (15, 60), "b": (100, 200)})  # only 'a' advanced
        self.tr.poll_once()
        a = self.tr.data["models"]["a"]
        b = self.tr.data["models"]["b"]
        self.assertEqual(a["prompt"], 5)
        self.assertEqual(a["generated"], 40)
        self.assertEqual(b["prompt"], 0)          # 'b' was idle -> no delta
        self.assertEqual(b["generated"], 0)

    def test_unloaded_model_baseline_dropped(self):
        self._wire_multi({"a": (10, 20), "b": (0, 0)})
        self.tr.poll_once()                       # baseline both
        self._wire_multi({"a": (10, 20)})         # 'b' evicted
        self.tr.poll_once()
        self.assertNotIn("b", self.tr._prev)
        # 'b' reloads later: fresh baseline, no stale diff attributed
        self._wire_multi({"a": (10, 20), "b": (500, 500)})
        self.tr.poll_once()
        self.assertEqual(self.tr.data["models"]["b"]["prompt"], 0)

    def test_counter_reset_guard_still_holds(self):
        self._wire_multi({"a": (10, 20)})
        self.tr.poll_once()
        self._wire_multi({"a": (2, 3)})           # router restarted -> counters reset
        self.tr.poll_once()
        self.assertEqual(self.tr.data["models"]["a"]["prompt"], 0)
        self.assertEqual(self.tr.data["models"]["a"]["generated"], 0)


if __name__ == "__main__":
    unittest.main()
