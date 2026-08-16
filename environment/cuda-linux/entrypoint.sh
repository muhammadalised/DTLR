#!/usr/bin/env bash
set -euo pipefail

native_root="${DTLR_NATIVE_ROOT:-/workspace/dtlr/.native}"
if ! PYTHONPATH="${native_root}:${PYTHONPATH:-}" python - <<'PY'
import MultiScaleDeformableAttention
import sort_vertices
PY
then
  echo "Building DTLR CUDA extensions for this mounted checkout..." >&2
  bash environment/cuda-linux/build_extensions.sh
fi

export PYTHONPATH="${native_root}:/workspace/dtlr:${PYTHONPATH:-}"
exec "$@"
