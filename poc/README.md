# DTLR → TVA bigram POC (August 28)

This directory is an additive adapter around upstream DTLR. It does not change
the model, decoding, IAM splits, or threshold/NMS behavior. The initial scope is
IAM line images and within-token adjacent character pairs (pairs touching
whitespace are excluded).

## Scientific contract

- Character identity comes from the IAM ground-truth transcription.
- Approximate location comes from ordered, thresholded, NMS-filtered DTLR boxes.
- A deterministic Levenshtein alignment maps transcription positions to boxes.
  Substitutions remain visible in `left_alignment`/`right_alignment`; exact-only
  aggregate scores are reported separately. Missing boxes make a pair unusable.
- Dark pixels are binarized with Otsu's threshold (or an explicitly recorded
  fixed threshold) and labelled with 8-connectivity.
- A pair is `connected` when one physical ink component intersects both aligned
  character boxes. This operational definition must be validated visually before
  it is treated as a handwriting boundary label.
- Aggregates are keyed by dataset **and split**. Do not pool train/validation/test
  evidence or tune TVA using held-out IAM evidence.

## Checkpoints are not interchangeable

`english-pretrained` means synthetic English language pretraining and is an
initialization checkpoint for IAM fine-tuning. `iam-finetuned` means a checkpoint
after the repository's IAM fine-tuning procedure. The August 28 localization
milestone is verified only by an **IAM-finetuned** checkpoint. Every detection
record contains the declared kind and SHA-256 digest; preserve the original
download filename and source URL in the run notes.

The same distinction will later apply to READ: `german-pretrained` is not a
`read-finetuned` checkpoint. READ is intentionally out of this first milestone.

## Expected external layout

The compose mounts expose these paths inside the container:

```text
/mnt/dtlr-data/
  IAM_new/labels.pkl
  IAM_new/data/imgs/lines/*.jpg
/mnt/dtlr-weights/
  finetuned/IAM/checkpoint.pth
/mnt/dtlr-output/
```

The upstream repository already tracks a small IAM `labels.pkl`; prefer the copy
distributed with the exact image preprocessing used for the run and record its
hash. Dataset images, checkpoints, detections, and exports must stay outside Git.

## Milestone run on Linux/WSL + RTX 4060

First follow [the CUDA environment guide](../environment/cuda-linux/README.md).
Then run a small, deterministic slice:

```bash
docker compose --env-file .env -f environment/cuda-linux/compose.yaml run --rm dtlr \
  python poc/scripts/export_iam_detections.py \
  --data-root /mnt/dtlr-data \
  --checkpoint /mnt/dtlr-weights/finetuned/IAM/checkpoint.pth \
  --checkpoint-kind iam-finetuned \
  --split test --start 0 --limit 8 --threshold 0.3 --nms 0.5 \
  --output /mnt/dtlr-output/iam-test-small/detections.jsonl

docker compose --env-file .env -f environment/cuda-linux/compose.yaml run --rm dtlr \
  python poc/scripts/build_bigram_evidence.py \
  --detections /mnt/dtlr-output/iam-test-small/detections.jsonl \
  --data-root /mnt/dtlr-data \
  --output-dir /mnt/dtlr-output/iam-test-small/bigrams
```

The second command writes row-level evidence and split-specific aggregates in
both CSV and JSON, plus a manifest. Before scaling up, inspect box overlays and
CCL assignments for the eight lines. That visualization/acceptance step remains
an explicit milestone gate; the current exporter deliberately does not silently
invent acceptance criteria.

## Local tests (no CUDA required)

```bash
PYTHONPATH=poc python -m unittest discover -s poc/tests -v
```

## Provenance checklist

For each retained run, save outside Git:

- repository commit and `git diff --stat` (a dirty tree must be stated);
- container image ID/digest, `pip freeze`, NVIDIA driver, and GPU name;
- checkpoint kind, original filename/source, and SHA-256;
- IAM archive/preprocessing provenance and hashes for `labels.pkl` and the line list;
- split, start/IDs, thresholds, NMS, binarization threshold, and output hashes;
- manual visual-QA decision and any excluded lines, with reasons.
