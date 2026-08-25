import conftest_paths  # noqa: F401
import os, tempfile, threading, unittest
import hub


class TestQueue(unittest.TestCase):
    """The DownloadManager accepts multiple start() calls by queueing them
    FIFO and draining them sequentially (one at a time)."""

    def _dm(self):
        dm = hub.DownloadManager()
        # Replace _run with a controllable stub so we can observe queue order
        # without real network I/O.
        self.runs = []
        self.started = threading.Event()

        def fake_run(repo, paths, dest_dir):
            self.runs.append((repo, tuple(paths), dest_dir))
            self.started.set()
            while not dm.state.get("cancel"):
                threading.Event().wait(0.01)
            raise hub.Cancelled()

        dm._run = fake_run
        return dm

    def test_second_start_queues_instead_of_rejecting(self):
        dm = self._dm()
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(dm.start("a/b", ["x.gguf"], d))
            self.assertTrue(dm.start("c/d", ["y.gguf"], d))  # queued, not rejected
            self.started.wait(5)
            self.assertEqual(len(self.runs), 1)      # only first is running
            self.assertEqual(dm.state["queued"], 1)  # one pending
            dm.cancel()

    def test_queue_drains_in_fifo_order(self):
        dm = hub.DownloadManager()
        order = []
        lock = threading.Lock()

        def fake_run(repo, paths, dest_dir):
            with lock:
                order.append(repo)
            # immediately "finish" so the next job is picked up
            dm.state.update(phase="done")

        dm._run = fake_run
        with tempfile.TemporaryDirectory() as d:
            dm.start("repo/one", ["a.gguf"], d)
            dm.start("repo/two", ["b.gguf"], d)
            dm.start("repo/three", ["c.gguf"], d)
            # wait for the worker loop to drain all three
            for _ in range(200):
                if not dm.state["running"]:
                    break
                threading.Event().wait(0.02)
            self.assertEqual(order, ["repo/one", "repo/two", "repo/three"])

    def test_duplicate_job_is_deduplicated(self):
        dm = self._dm()
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(dm.start("a/b", ["x.gguf"], d))
            # same job again -> dedupe, does not grow the queue
            self.assertTrue(dm.start("a/b", ["x.gguf"], d))
            self.started.wait(5)
            self.assertEqual(dm.state["queued"], 0)
            dm.cancel()


if __name__ == "__main__":
    unittest.main()


class TestRemoveQueued(unittest.TestCase):
    def test_remove_queued_leaves_running_head(self):
        dm = hub.DownloadManager()
        # stub _run so the head job "runs" indefinitely
        started = threading.Event()

        def fake_run(repo, paths, dest_dir):
            started.set()
            while not dm.state.get("cancel"):
                threading.Event().wait(0.01)
            raise hub.Cancelled()

        dm._run = fake_run
        with tempfile.TemporaryDirectory() as d:
            dm.start("a/one", ["a.gguf"], d)   # running
            dm.start("b/two", ["b.gguf"], d)   # queued
            dm.start("c/three", ["c.gguf"], d) # queued
            started.wait(5)
            self.assertEqual(dm.state["queued"], 2)
            # remove the first queued job (b/two)
            self.assertTrue(dm.remove_queued("b/two", ["b.gguf"], d))
            self.assertEqual(dm.state["queued"], 1)
            # head (a/one) still running
            self.assertTrue(dm.state["running"])
            self.assertEqual(dm.state["repo"], "a/one")
            # removing the running head must be refused
            self.assertFalse(dm.remove_queued("a/one", ["a.gguf"], d))
            dm.cancel()

    def test_remove_queued_missing_returns_false(self):
        dm = hub.DownloadManager()
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(dm.remove_queued("nope/x", ["x.gguf"], d))


if __name__ == "__main__":
    unittest.main()
