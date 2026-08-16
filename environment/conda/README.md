# Conda environment for Linux/WSL + RTX 4060

This is the primary development environment for the IAM bigram POC. It keeps
DTLR separate from TVA while running directly in the same Linux/WSL host. The
environment uses Python 3.11.0, PyTorch 2.1.0+cu118, CUDA 11.8, and extensions
compiled for the RTX 4060's compute capability 8.9.

## Prerequisites

- Linux x86-64 or WSL2, not native Windows Python
- a current NVIDIA driver with `nvidia-smi` working in the Linux/WSL shell
- Conda or Miniforge

Check the host before creating the environment:

```bash
nvidia-smi
uname -m
```

`uname -m` must print `x86_64`. The Conda environment installs its own CUDA 11.8
toolkit, including `nvcc`; a newer system toolkit can remain installed.

## Create and verify

Run from the repository root:

```bash
conda env create -f environment/conda/environment.yml
conda activate dtlr-poc
source environment/conda/activate.sh
bash environment/conda/setup.sh
```

The setup installs pinned Python packages, compiles both custom CUDA extensions
into the ignored `.native/` directory, and runs the GPU preflight. A rebuild is
needed after changing Python, PyTorch, CUDA, extension sources, or GPU architecture:

```bash
rm -rf .native  # only this generated cache
source environment/conda/activate.sh
bash environment/cuda-linux/build_extensions.sh
```

If the environment already existed before CUDA 11.8 was added to this file,
update it before rebuilding:

```bash
conda env update -n dtlr-poc -f environment/conda/environment.yml --prune
conda activate dtlr-poc
source environment/conda/activate.sh
which nvcc
nvcc --version  # must report release 11.8
rm -rf .native  # remove only the failed/generated native cache
bash environment/conda/setup.sh
```

## Data and output paths

Copy `.env.example` to `.env`, use absolute Linux paths, and load it after
activating the environment:

```bash
cp .env.example .env
# edit .env
set -a
source .env
set +a
python environment/cuda-linux/preflight.py --require-gpu
```

For WSL2, paths under `/home/...` perform better than `/mnt/c/...` for this
workload. Dataset images, checkpoints, and outputs remain outside the repository.

Docker remains available under `environment/cuda-linux/` as an optional isolated
fallback; it is not required for this workflow.
