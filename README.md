# Kimodo motion generation

Generate text-directed motion with the public
[NVIDIA Kimodo source](https://github.com/nv-tlabs/kimodo), using Python and
PyTorch. Outputs are motion packages for preview and retargeting in the
[Workspots motion library](https://github.com/ghostline77/workspots).

## Setup

Use Python 3.12 or newer. Install upstream Kimodo and a GPU-compatible PyTorch
environment using the upstream installation guide. Keep its source checkout and
downloaded checkpoints outside this repository.

Run the commands below in that Python environment, from this repository.
Replace `path/to/kimodo` and `path/to/checkpoints` with local directories.

Build the native cleanup dependency with CMake and a C++17 compiler available:

```powershell
python -m pip install --no-deps --no-build-isolation --target work/native-deps path/to/kimodo/MotionCorrection
```

The generator checks its dependencies and local model caches before inference.
It runs offline; missing model files produce an error.

## Generate a clip

```powershell
python kimodo_motion.py "A person stands and waves hello with their right hand." --upstream path/to/kimodo --checkpoint-root path/to/checkpoints --seconds 4 --seed 42 --output work/wave-42
```

Choose a new output directory for each run. Import the resulting `motion.json`
through **Import motion** in Workspots. Review the source and retargeted
animation before using it in a production asset.

See [generation settings and outputs](docs/generation.md) for memory modes,
embedding reuse and batches. Generated packages contain source motion and
provenance; they are not game-ready Cyberpunk `.anims` resources.
