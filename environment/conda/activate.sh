#!/usr/bin/env bash
# Source this file after activating the Conda environment:
#   source environment/conda/activate.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This file must be sourced, not executed." >&2
  exit 2
fi

dtlr_activate_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DTLR_REPO_ROOT="$(cd "${dtlr_activate_dir}/../.." && pwd)"
export DTLR_NATIVE_ROOT="${DTLR_REPO_ROOT}/.native"
export PYTHONPATH="${DTLR_NATIVE_ROOT}:${DTLR_REPO_ROOT}:${PYTHONPATH:-}"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
  export CUDA_HOME="${CONDA_PREFIX}"
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
else
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
fi
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
unset dtlr_activate_dir
