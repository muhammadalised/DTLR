#!/usr/bin/env python3
"""Prepare only the raw IAM lines named by a frozen selection manifest."""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


PYLAIA_REFERENCE = {
    "repository": "https://github.com/carmocca/PyLaia-examples.git",
    "commit": "2151ec611558fa2b5eaefe443f58bb6df8b6027f",
    "script": "iam-htr/src/prepare_images.sh",
}


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tool_version(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    text = (result.stdout or result.stderr).strip()
    return text.splitlines()[0] if text else f"exit={result.returncode}"


def find_raw_line(raw_root: Path, line_id: str) -> Path:
    matches = sorted(path for path in raw_root.rglob(f"{line_id}.png") if path.is_file())
    if not matches:
        raise FileNotFoundError(f"raw IAM line not found: {line_id}.png under {raw_root}")
    if len(matches) > 1:
        joined = "\n  ".join(str(path) for path in matches)
        raise RuntimeError(f"multiple raw IAM lines found for {line_id}:\n  {joined}")
    return matches[0]


def process_image(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=target.parent, suffix=".jpg", delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        enhancer = subprocess.Popen(
            ["imgtxtenh", "-d", "118.110", str(source), "png:-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert enhancer.stdout is not None
        converter = subprocess.Popen(
            [
                "convert", "png:-", "-deskew", "40%",
                "-bordercolor", "white", "-border", "5", "-trim",
                "-bordercolor", "white", "-border", "20x0", "+repage",
                "-strip", f"jpg:{temporary}",
            ],
            stdin=enhancer.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        enhancer.stdout.close()
        _, converter_stderr = converter.communicate()
        enhancer_stderr = enhancer.stderr.read() if enhancer.stderr else b""
        enhancer_returncode = enhancer.wait()
        if enhancer_returncode or converter.returncode:
            raise RuntimeError(
                f"preprocessing failed for {source}\n"
                f"imgtxtenh: {enhancer_stderr.decode(errors='replace')}\n"
                f"convert: {converter_stderr.decode(errors='replace')}"
            )
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"preprocessing produced an empty image for {source}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--raw-lines-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    for command in ("imgtxtenh", "convert"):
        if not shutil.which(command):
            raise SystemExit(f"required command is unavailable: {command}")

    selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    if selection.get("schema_version") != "dtlr.iam-selection.v1":
        raise SystemExit("unsupported selection manifest")
    selected_ids = [row["id"] for row in selection["lines"]]

    # Resolve every source before writing anything, so a partial raw archive
    # cannot silently produce a partial validation sample.
    sources = {line_id: find_raw_line(args.raw_lines_root, line_id) for line_id in selected_ids}
    output_root = args.data_root / "IAM_new/data/imgs/lines"
    records = []
    for index, line_id in enumerate(selected_ids, 1):
        source = sources[line_id]
        target = output_root / f"{line_id}.jpg"
        status = "existing"
        if not target.exists():
            process_image(source, target)
            status = "created"
        records.append({
            "line_id": line_id,
            "status": status,
            "source_relpath": str(source.relative_to(args.raw_lines_root)),
            "source_sha256": sha256_file(source),
            "output_relpath": str(target.relative_to(args.data_root)),
            "output_sha256": sha256_file(target),
        })
        print(f"[{index}/{len(selected_ids)}] {line_id}: {status}")

    manifest = {
        "schema_version": "dtlr.iam-preprocessing.v1",
        "dataset": "IAM",
        "selection_manifest_sha256": sha256_file(args.selection_manifest),
        "reference": PYLAIA_REFERENCE,
        "operation": {
            "imgtxtenh": ["-d", "118.110", "INPUT", "png:-"],
            "imagemagick": [
                "png:-", "-deskew", "40%", "-bordercolor", "white",
                "-border", "5", "-trim", "-bordercolor", "white",
                "-border", "20x0", "+repage", "-strip", "OUTPUT.jpg",
            ],
            "height_resize_applied": False,
        },
        "tools": {
            "imgtxtenh_path": shutil.which("imgtxtenh"),
            "imagemagick_path": shutil.which("convert"),
            "imagemagick_version": tool_version(["convert", "-version"]),
        },
        "records": records,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote preprocessing manifest: {args.output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
