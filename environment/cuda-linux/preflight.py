#!/usr/bin/env python3
import argparse
import json
import os
import platform
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the supported DTLR CUDA runtime")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    import torch

    report = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "repo_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    errors = []
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        errors.append("DTLR CUDA runtime requires Linux x86_64")
    if platform.python_version() != "3.11.0":
        errors.append("expected Python 3.11.0")
    if not torch.__version__.startswith("2.1.0"):
        errors.append("expected PyTorch 2.1.0")
    if torch.version.cuda != "11.8":
        errors.append("expected PyTorch CUDA runtime 11.8")
    if args.require_gpu and not torch.cuda.is_available():
        errors.append("CUDA GPU is not visible; check the NVIDIA driver and CUDA setup")
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
        report["compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        try:
            import MultiScaleDeformableAttention  # noqa: F401
            report["ms_deform_attn"] = "imported"
        except Exception as exc:
            errors.append(f"MultiScaleDeformableAttention import failed: {exc}")
        try:
            import sort_vertices  # noqa: F401
            report["sort_vertices"] = "imported"
        except Exception as exc:
            errors.append(f"sort_vertices import failed: {exc}")

    for key in ("DTLR_DATA_ROOT", "DTLR_WEIGHTS_ROOT", "DTLR_OUTPUT_ROOT"):
        report[key] = os.environ.get(key)
        if args.require_gpu and not report[key]:
            errors.append(f"{key} is not set")
    report["errors"] = errors
    print(json.dumps(report, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
