"""Run public Kimodo with bounded CPU BLAS and staged text-model residency."""
from __future__ import annotations

import ctypes
import gc
import importlib
import json
import os
from pathlib import Path
import time


def memory_snapshot() -> dict:
    if os.name != "nt":
        return {}
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ("peak_ws", "ws", "peak_paged", "paged",
                "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile", "private")]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    api = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    api.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    values = Counters()
    values.cb = ctypes.sizeof(values)
    if not api(kernel.GetCurrentProcess(), ctypes.byref(values), values.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"private_bytes": values.private, "working_set_bytes": values.ws,
            "peak_private_bytes": values.peak_pagefile, "peak_working_set_bytes": values.peak_ws}


def configure_blas(threads: int) -> list[dict]:
    import torch
    torch.set_num_threads(threads)
    rows = []
    if os.name == "nt":
        directory = Path(torch.__file__).parent.parent / "_rocm_sdk_core/bin"
        for name in ("rocm-openblas.dll", "rocm-openblas64.dll"):
            path = directory / name
            if not path.exists():
                continue
            library = ctypes.CDLL(str(path))
            library.openblas_set_num_threads.argtypes = [ctypes.c_int]
            library.openblas_set_num_threads.restype = None
            library.openblas_get_num_threads.restype = ctypes.c_int
            before = library.openblas_get_num_threads()
            library.openblas_set_num_threads(threads)
            after = library.openblas_get_num_threads()
            if after != threads:
                raise RuntimeError(f"Could not limit {name} to {threads} threads")
            rows.append({"library": str(path), "before": before, "after": after})
    return rows


class CachedTextEncoder:
    """Small CPU embeddings survive after the Llama model has been released."""
    llm_dim = 4096

    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __call__(self, texts):
        import torch
        single = isinstance(texts, str)
        values = [texts] if single else texts
        tensors = []
        for text in values:
            if not text.strip():
                tensors.append(torch.zeros(1, 1, self.llm_dim))
            elif text in self.embeddings:
                tensors.append(self.embeddings[text])
            else:
                raise ValueError(f"Prompt was not prepared before releasing Llama: {text!r}")
        result = torch.cat(tensors, dim=0)
        lengths = [1] * len(values)
        return (result[0], lengths[0]) if single else (result, lengths)


def enable_sequential_offload(encoder, device):
    import torch
    from accelerate import cpu_offload
    from kimodo.model.llm2vec.llm2vec import LLM2Vec

    class SequentialLLM2Vec(LLM2Vec):
        def to(self, target, *args, **kwargs):
            # Upstream encode() calls to(device) before every forward. Moving all
            # parameters there would defeat Accelerate's layer offload hooks.
            if args or kwargs or torch.device(target) != torch.device(device):
                raise ValueError("An offloaded text encoder cannot be moved as a whole")
            return self

    source = encoder.model
    encoder.model = SequentialLLM2Vec(source.model, source.tokenizer,
        pooling_mode=source.pooling_mode, max_length=source.max_length,
        doc_max_length=source.doc_max_length, skip_instruction=source.skip_instruction)
    cpu_offload(encoder.model, execution_device=torch.device(device), offload_buffers=True)
    encoder._device = device


def main():
    import torch
    from kimodo.scripts import generate
    from kimodo.sanitize import sanitize_texts
    from kimodo.model.loading import instantiate_from_dict
    loading = importlib.import_module("kimodo.model.load_model")
    # Fail before loading Llama if the required native cleanup module is absent.
    import motion_correction

    report = Path(os.environ["GHOSTLINE_TEXT_REPORT"])
    phases = []
    def record(stage, **details):
        row = {"stage": stage, "time": time.time(), **memory_snapshot(), **details}
        if torch.cuda.is_available():
            row.update(gpu_allocated_bytes=torch.cuda.memory_allocated(),
                       gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        phases.append(row)
        report.write_text(json.dumps(phases, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)

    record("blas_configured", libraries=configure_blas(int(os.environ.get("GHOSTLINE_BLAS_THREADS", "1"))))
    args = generate.parse_args()
    if "soma" not in args.model.lower():
        raise ValueError("This wrapper currently supports SOMA models at 30 FPS")
    inputs = generate.get_generation_inputs(args, 30)
    prompts = list(dict.fromkeys(sanitize_texts(inputs["texts"])))
    mode = os.environ.get("GHOSTLINE_TEXT_MODE", "sequential")
    if mode != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU text mode requires a CUDA or ROCm PyTorch device")
    reused = os.environ.get("GHOSTLINE_TEXT_EMBEDDINGS")
    if reused:
        from safetensors import safe_open
        with safe_open(reused, framework="pt", device="cpu") as source:
            metadata = source.metadata()
            if json.loads(metadata["prompts"]) != prompts or metadata["mode"] != mode:
                raise ValueError("Cached embedding prompts or execution mode do not match")
            embeddings = {prompt: source.get_tensor(str(i)) for i, prompt in enumerate(prompts)}
        if any(value.shape != (1, 1, 4096) or not torch.isfinite(value).all()
               for value in embeddings.values()):
            raise ValueError("Invalid cached embedding")
        record("text_embeddings_reused", prompts=len(embeddings))
    else:
        # Keep the exact upstream LLM2Vec model, precision and two-adapter loading.
        os.environ["TEXT_ENCODER_DEVICE"] = "cpu"
        encoder = instantiate_from_dict(loading._build_local_text_encoder_conf())
        record("text_loaded_cpu")
        if mode == "sequential":
            enable_sequential_offload(encoder, "cuda:0")
        elif mode == "gpu-staged":
            encoder.to("cuda:0")
        record("text_ready", mode=mode)
        embeddings = {}
        with torch.inference_mode():
            for prompt in prompts:
                value, _ = encoder([prompt])
                if not torch.isfinite(value).all():
                    raise ValueError("Non-finite text embedding")
                embeddings[prompt] = value.detach().cpu()
        record("text_encoded", prompts=len(embeddings))
        del encoder
    from safetensors.torch import save_file
    save_file({str(index): embeddings[prompt].contiguous() for index, prompt in enumerate(prompts)},
              str(report.with_name("text-embeddings.safetensors")),
              metadata={"prompts": json.dumps(prompts), "mode": mode})
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    record("text_released_before_motion")
    cached = CachedTextEncoder(embeddings)
    def load_with_embeddings(*values, **kwargs):
        loaded = loading.load_model(*values, **kwargs, text_encoder=cached)
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        record("motion_loaded", device=str(model.device))
        return loaded
    generate.load_model = load_with_embeddings
    generate.main()
    record("generation_complete")


if __name__ == "__main__":
    main()
