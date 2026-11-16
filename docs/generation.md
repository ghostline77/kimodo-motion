# Generation settings and outputs

Run commands from the Kimodo motion repository in the
[configured Python environment](../README.md#setup).

## Memory modes

| `--text-mode` | Text encoder execution |
| --- | --- |
| `sequential` (default) | Keep weights in host memory and transfer modules to the GPU for computation. |
| `cpu` | Compute text embeddings on the CPU. |
| `gpu-staged` | Load the complete text encoder on the GPU. |

The text encoder is released before the motion model loads. Modes do not switch
automatically after a failure. `--blas-threads` defaults to 1 and applies to this
process. `--native-deps` defaults to `work/native-deps` beside the generator.

Use `--seconds`, `--seed` and `--steps` to set duration, random seed and
inference steps. Defaults are 4 seconds, seed 42 and 50 steps.
`--model` defaults to `Kimodo-SOMA-RP-v1.1`.

## Reuse prompt embeddings

Pass `--text-embeddings path/to/text-embeddings.safetensors` to reuse saved
conditioning after a later generation failure. The wrapper checks the prompt,
mode, tensor dimensions and finite values, and records the source hash.

## Generate a batch

```powershell
python kimodo_test_batch.py --plan examples/retarget-stress.json --output work/retarget-stress --upstream path/to/kimodo --checkpoint-root path/to/checkpoints
```

The destination must be new. The batch encodes all prompts in one text-model
load, releases that model, then generates independent clips with one shared
motion-model load. Each job retains its own duration and seed. Clips are not
stitched into a continuous performance.

## Output reference

| File | Content |
| --- | --- |
| `motion.json` | Hash-validated import package |
| `motion.npz` | Source motion arrays |
| `motion.bvh` | SOMA-77 motion with a standard T-pose rest skeleton |
| `request.json` | Prompt, settings and source/checkpoint identity |
| `generation.log` | Inference diagnostics |
| `text-memory.json` | Memory measurements by phase |

Runner sources and provenance accompany each package. Batch receipts distinguish
completed outputs from failed jobs. Only finite motions with valid rotations and
the requested frame count are packaged.

Motion exports use 30 FPS. The checkpoint generates a compact 30-joint SOMA
skeleton; upstream expands it to 77 joints using a relaxed hand template. It does
not generate independent finger articulation. Retargeting, native game-resource
encoding and runtime animation validation are separate steps.

## Development checks

These checks do not generate a production clip. The runtime checks use upstream
Kimodo on the Python import path.

```powershell
python -m unittest -v test_runtime_bootstrap.py
python -m unittest -v test_kimodo_test_batch.py
```
