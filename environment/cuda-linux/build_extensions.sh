#!/usr/bin/env bash
set -euo pipefail

case "$(uname -s)/$(uname -m)" in
  Linux/x86_64) ;;
  *) echo "Refusing CUDA extension build on $(uname -s)/$(uname -m); use Linux x86_64." >&2; exit 2 ;;
esac

nvcc_path="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
if [[ ! -x "${nvcc_path}" ]]; then
  echo "CUDA compiler not found at ${nvcc_path}. Activate the Conda environment helper first." >&2
  exit 2
fi
nvcc_version="$(${nvcc_path} --version)"
if [[ "${nvcc_version}" != *"release 11.8"* ]]; then
  echo "Refusing CUDA extension build: ${nvcc_path} is not CUDA 11.8." >&2
  echo "${nvcc_version}" >&2
  exit 2
fi

python - <<'PY'
import torch
assert torch.__version__.startswith("2.1.0"), torch.__version__
assert torch.version.cuda == "11.8", torch.version.cuda
assert torch.cuda.is_available(), "extension compilation requires a GPU-visible runtime"
PY

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
native_root="${DTLR_NATIVE_ROOT:-${repo_root}/.native}"
mkdir -p "${native_root}/tmp/msda" "${native_root}/tmp/sort_vertices"

(
  cd "${repo_root}/models/dino/ops"
  python setup.py build_ext \
    --build-lib "${native_root}" \
    --build-temp "${native_root}/tmp/msda"
)
(
  cd "${repo_root}/models/dino/cuda_op"
  python setup.py build_ext \
    --build-lib "${native_root}" \
    --build-temp "${native_root}/tmp/sort_vertices"
)
