# Optional Docker CUDA environment

The primary native workflow is documented in
[`environment/conda/README.md`](../conda/README.md). This Docker variant is an
optional isolation fallback using Linux x86-64 (native or WSL2), NVIDIA driver
compatible with CUDA 11.8, Python 3.11.0, PyTorch 2.1.0+cu118, and an RTX 4060
(compute capability 8.9). Do not build the CUDA extensions on Apple Silicon.

## Host preparation (RTX 4060)

Install a current NVIDIA driver, Docker, and NVIDIA Container Toolkit. On WSL2,
install the NVIDIA Windows driver and Docker Desktop with WSL integration; do not
install a second Linux display driver inside WSL. Verify `nvidia-smi` works from
the target Linux/WSL shell before continuing.

```bash
cp .env.example .env
# edit the three absolute host paths in .env
docker compose --env-file .env -f environment/cuda-linux/compose.yaml build
docker compose --env-file .env -f environment/cuda-linux/compose.yaml run --rm dtlr \
  python environment/cuda-linux/preflight.py --require-gpu
```

The first GPU-backed run compiles the native operators into the ignored
`.native/` cache in the mounted checkout. Compilation is intentionally not done
during `docker build`, because standard Docker builds do not expose the GPU and
upstream DTLR's setup script refuses that unsupported condition.

The repository is bind-mounted at runtime, but dataset and weight roots are
separate mounts. `DTLR_DATA_ROOT` is read-only, `DTLR_WEIGHTS_ROOT` is read-only,
and outputs are written only under `DTLR_OUTPUT_ROOT`.

The container image is reproducible at the version-pin level. For archival runs,
record the resolved image digest and `pip freeze` alongside the run manifest.
