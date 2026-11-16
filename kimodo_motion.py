"""Use public Kimodo/PyTorch, retaining portable source motion and provenance."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_motion(path: Path, expected_frames: int | None = None) -> dict:
    """Reject invalid or ambiguous output before making it available for import."""
    with np.load(path, allow_pickle=False) as motion:
        positions = motion["posed_joints"]
        if positions.ndim != 3 or positions.shape[1:] != (77, 3):
            raise ValueError("Expected one SOMA-77 motion [frames,77,3]")
        frames = positions.shape[0]
        if frames < 1 or expected_frames not in (None, frames):
            raise ValueError(f"Unexpected frame count {frames}; expected {expected_frames}")
        for name in ("local_rot_mats", "global_rot_mats"):
            rotations = motion[name]
            if rotations.shape != (frames, 77, 3, 3):
                raise ValueError(f"Invalid {name} shape")
            identity = rotations @ np.swapaxes(rotations, -1, -2)
            if not np.allclose(identity, np.eye(3), atol=0.01):
                raise ValueError(f"Invalid {name}: non-orthogonal rotations")
            if not np.allclose(np.linalg.det(rotations), 1, atol=0.01):
                raise ValueError(f"Invalid {name}: reflected rotations")
        for name in motion.files:
            value = motion[name]
            if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                raise ValueError(f"Invalid values in {name}")
    return {"frames": frames, "fps": 30, "duration_seconds": frames / 30,
            "skeleton": "soma77", "coordinate_system": "kimodo-native-metres",
            "sha256": digest(path)}


def generate(args: argparse.Namespace) -> None:
    upstream = args.upstream.resolve(strict=True)
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip():
        raise ValueError("Use an unmodified upstream checkout for reproducible generation")
    if not 2 <= args.seconds <= 10 or not np.isfinite(args.seconds):
        raise ValueError("Duration must be between two and ten seconds")
    frames = round(args.seconds * 30)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runner = Path(__file__).resolve()
    (output / "runner.py").write_bytes(runner.read_bytes())
    checkpoint = args.checkpoint_root.resolve(strict=True) / args.model
    checkpoint_hashes = {str(p.relative_to(checkpoint)): digest(p) for p in
                         (checkpoint / "config.yaml", checkpoint / "model.safetensors")}
    command = [sys.executable, "-m", "kimodo.scripts.generate", args.prompt,
               "--model", args.model, "--duration", str(frames / 30),
               "--num_samples", "1", "--seed", str(args.seed),
               "--diffusion_steps", str(args.steps), "--bvh", "--bvh_standard_tpose",
               "--output", str(output / "motion")]
    helper = Path(__file__).with_name("runtime_bootstrap.py")
    executed_helper = output / "runtime_bootstrap.py"
    executed_helper.write_bytes(helper.read_bytes())
    command = [sys.executable, str(executed_helper), *command[3:]]
    native_deps = args.native_deps.resolve()
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(upstream), str(native_deps)]), CHECKPOINT_DIR=str(checkpoint.parent),
               TEXT_ENCODERS_DIR=str(checkpoint.parent.parent),
               TEXT_ENCODER_MODE="local", TEXT_ENCODER_DEVICE="cpu",
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    env["GHOSTLINE_TEXT_REPORT"] = str(output / "text-memory.json")
    env["GHOSTLINE_TEXT_MODE"] = args.text_mode
    env["GHOSTLINE_BLAS_THREADS"] = str(args.blas_threads)
    if args.text_embeddings:
        env["GHOSTLINE_TEXT_EMBEDDINGS"] = str(args.text_embeddings.resolve(strict=True))
    started = time.time()
    request = {"prompt": args.prompt, "seed": args.seed, "frames": frames,
               "model": args.model, "steps": args.steps, "engine": "upstream-pytorch",
               "upstream_revision": revision, "checkpoint_hashes": checkpoint_hashes,
               "command": command, "text_mode": args.text_mode, "started": started,
               "working_directory": str(checkpoint.parent.parent), "runner_sha256": digest(runner)}
    request["runtime_bootstrap_sha256"] = digest(executed_helper)
    request["blas_threads"] = args.blas_threads
    if args.text_embeddings:
        request["reused_text_embeddings"] = {"path": env["GHOSTLINE_TEXT_EMBEDDINGS"],
            "sha256": digest(args.text_embeddings)}
    request["packages"] = {name: importlib.metadata.version(name) for name in
        ("torch", "transformers", "peft", "accelerate", "numpy", "safetensors")}
    (output / "request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
    with (output / "generation.log").open("w", encoding="utf-8") as log:
        # Public adapter configs reference their base as meta-llama/<model>.
        # Resolve that relative path within the external checkpoint root.
        result = subprocess.run(command, cwd=checkpoint.parent.parent, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        (output / "failure.json").write_text(json.dumps({"exit_code": result.returncode,
            "elapsed_seconds": time.time() - started}), encoding="utf-8")
        raise RuntimeError(f"Kimodo exited {result.returncode}; see {output / 'generation.log'}")
    inspected = inspect_motion(output / "motion.npz", frames)
    package = {"format": "ghostline-generated-motion", "version": 1,
               **request, **inspected, "review_status": "unreviewed",
               "retargeted": False, "runtime_verified": False,
               "elapsed_seconds": time.time() - started,
               "files": {name: digest(output / name) for name in ("motion.npz", "motion.bvh", "generation.log")}}
    (output / "motion.json").write_text(json.dumps(package, indent=2), encoding="utf-8")
    print(json.dumps(package, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--model", default="Kimodo-SOMA-RP-v1.1")
    parser.add_argument("--seconds", type=float, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--text-mode", choices=("cpu", "sequential", "gpu-staged"), default="sequential")
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--native-deps", type=Path, default=Path(__file__).parent / "work/native-deps")
    parser.add_argument("--text-embeddings", type=Path, help="Reuse a saved, matching prompt embedding")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000:
        parser.error("steps must be in 1..1000")
    if not 1 <= args.blas_threads <= 16:
        parser.error("blas-threads must be in 1..16")
    generate(args)


if __name__ == "__main__":
    main()
