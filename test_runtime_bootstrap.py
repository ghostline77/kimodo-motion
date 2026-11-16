import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
import unittest
from types import SimpleNamespace

import torch
from runtime_bootstrap import CachedTextEncoder, configure_blas, enable_sequential_offload, memory_snapshot


class RuntimeTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "Windows commit regression")
    def test_cpu_blas_does_not_commit_multiple_gigabytes_for_small_gemm(self):
        configured = configure_blas(1)
        for library in configured:
            self.assertEqual(library["after"], 1)
        before = memory_snapshot()["private_bytes"]
        result = torch.randn(4096, 16) @ torch.randn(16, 4096)
        extra = memory_snapshot()["private_bytes"] - before
        self.assertEqual(result.shape, (4096, 4096))
        self.assertLess(extra, 1024 ** 3, f"Small GEMM committed {extra / 1024**3:.2f} GiB")

    def test_cached_embeddings_preserve_order_and_reject_unknown_prompts(self):
        cached = CachedTextEncoder({"wave": torch.ones(1, 1, 4096)})
        result, lengths = cached(["", "wave", "wave"])
        self.assertEqual(lengths, [1, 1, 1])
        self.assertEqual(result.shape, (3, 1, 4096))
        self.assertEqual(result[0].count_nonzero(), 0)
        self.assertTrue(torch.equal(result[1], result[2]))
        with self.assertRaises(ValueError):
            cached(["unprepared"])

    @unittest.skipUnless(torch.cuda.is_available(), "requires a GPU")
    def test_sequential_offload_computes_on_gpu_without_whole_model_move(self):
        from kimodo.model.llm2vec.llm2vec import LLM2Vec
        configure_blas(1)
        model = torch.nn.Linear(4, 4, bias=False)
        model.config = SimpleNamespace()
        with torch.no_grad():
            model.weight.copy_(torch.eye(4) * 3)
        encoder = SimpleNamespace(model=LLM2Vec(model, None), _device="cpu")
        enable_sequential_offload(encoder, "cuda:0")
        encoder.model.to("cuda:0")  # Upstream encode() calls this every time.
        with torch.inference_mode():
            value = encoder.model.model(torch.ones(1, 4, device="cuda:0"))
        self.assertEqual(value.device.type, "cuda")
        self.assertTrue(torch.equal(value.cpu(), torch.ones(1, 4) * 3))
        self.assertEqual(encoder.model.model.weight.device.type, "meta")
        with self.assertRaises(ValueError):
            encoder.model.to("cpu")


if __name__ == "__main__":
    unittest.main()
