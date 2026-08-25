import conftest_paths  # noqa: F401
import unittest
import scanner


class TestShardSlug(unittest.TestCase):
    def test_shard_suffix_stripped_from_id(self):
        # A 3-shard model should collapse to a clean id without -00001-of-00003.
        paths = [
            "/m/models/unsloth--Qwen3.5-122B-A10B-GGUF/Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf",
            "/m/models/unsloth--Qwen3.5-122B-A10B-GGUF/Qwen3.5-122B-A10B-UD-Q4_K_XL-00002-of-00003.gguf",
            "/m/models/unsloth--Qwen3.5-122B-A10B-GGUF/Qwen3.5-122B-A10B-UD-Q4_K_XL-00003-of-00003.gguf",
        ]
        entries = scanner.build_entries(paths)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["id"], "qwen3.5-122b-a10b-ud-q4-k-xl")

    def test_non_sharded_id_unchanged(self):
        paths = ["/m/models/Model-7B-Q8_0.gguf"]
        entries = scanner.build_entries(paths)
        self.assertEqual(entries[0]["id"], "model-7b-q8-0")


if __name__ == "__main__":
    unittest.main()
