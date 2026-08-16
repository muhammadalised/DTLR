#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

case "$(uname -s)/$(uname -m)" in
  Linux/x86_64) ;;
  *) echo "The DTLR CUDA environment requires Linux x86_64." >&2; exit 2 ;;
esac

python - <<'PY'
import platform
assert platform.python_version() == "3.11.0", platform.python_version()
PY

python -m pip install -r "${script_dir}/requirements.txt"

export DTLR_REPO_ROOT="${repo_root}"
export DTLR_NATIVE_ROOT="${repo_root}/.native"
export PYTHONPATH="${DTLR_NATIVE_ROOT}:${repo_root}:${PYTHONPATH:-}"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
  export CUDA_HOME="${CONDA_PREFIX}"
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
else
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
fi
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

cd "${repo_root}"
bash environment/cuda-linux/build_extensions.sh
python environment/cuda-linux/preflight.py --require-gpu

echo
echo "Setup passed. In each new shell run:"
echo "  conda activate dtlr-poc"
echo "  source environment/conda/activate.sh"
echo "  set -a; source .env; set +a"
