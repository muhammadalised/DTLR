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
- Full-box intersection (`box-intersection-v1`) is retained only as rejected
  provenance: visual QA showed false positives when one character's ink entered
  the overlap between adjacent boxes. Float-based `exclusive-core-v2` is also
  retained as rejected provenance because outward pixel rounding could
  reintroduce a one-column overlap. The primary `exclusive-core-v2.1` method
  removes horizontal box overlap and rounds the facing core boundaries inward,
  then marks a pair `connected` only when one component intersects both
  raster-disjoint cores. Pairs without ink in both cores are unusable. This
  revised operational definition still requires validation on non-test data
  before it is treated as a handwriting boundary label. The observed failures
  and method changes are
  recorded in
  [`provenance/connectivity-method-change-2026-08-21.md`](provenance/connectivity-method-change-2026-08-21.md).
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

Use absolute Linux/WSL paths outside the repository:

```text
~/dtlr-data/
  IAM_new/labels.pkl
  IAM_new/data/imgs/lines/*.jpg
~/dtlr-weights/
  finetuned/IAM/checkpoint.pth
~/dtlr-output/
```

The upstream repository already tracks a small IAM `labels.pkl`; prefer the copy
distributed with the exact image preprocessing used for the run and record its
hash. Dataset images, checkpoints, detections, and exports must stay outside Git.

## Milestone run on Linux/WSL + RTX 4060

First follow [the Conda environment guide](../environment/conda/README.md), then
activate the environment and load the external path configuration:

```bash
conda activate dtlr-poc
source environment/conda/activate.sh
set -a; source .env; set +a
```

Then run a small, deterministic slice:

```bash
python poc/scripts/export_iam_detections.py \
  --data-root "${DTLR_DATA_ROOT}" \
  --checkpoint "${DTLR_WEIGHTS_ROOT}/finetuned/IAM/checkpoint.pth" \
  --checkpoint-kind iam-finetuned \
  --split test --start 0 --limit 8 --threshold 0.3 --nms 0.5 \
  --output "${DTLR_OUTPUT_ROOT}/iam-test-small/detections.jsonl"

python poc/scripts/build_bigram_evidence.py \
  --detections "${DTLR_OUTPUT_ROOT}/iam-test-small/detections.jsonl" \
  --data-root "${DTLR_DATA_ROOT}" \
  --output-dir "${DTLR_OUTPUT_ROOT}/iam-test-small/bigrams-exclusive-core-v2-1"

python poc/scripts/render_iam_qa.py \
  --detections "${DTLR_OUTPUT_ROOT}/iam-test-small/detections.jsonl" \
  --data-root "${DTLR_DATA_ROOT}" \
  --output-dir "${DTLR_OUTPUT_ROOT}/iam-test-small/qa-exclusive-core-v2-1"
```

The second command writes row-level evidence and split-specific aggregates in
both CSV and JSON, plus a manifest. The third command writes one side-by-side
box/alignment and CCL overview per line, enlarged pair-level crops under
`qa/pairs/`, a browser-friendly `qa/index.html`, and a `qa_manifest.json`. In a
connected pair crop the exact component spanning both exclusive cores is
magenta. For a disconnected pair, components intersecting only the left core are
blue and those intersecting only the right core are orange; box overlap is
omitted and unrelated ink is gray. The rejected `core-v2` and `box-v1` results
remain visible for comparison. Before scaling up,
manually inspect the crops and record an acceptance decision. This
visualization/acceptance step remains an explicit milestone gate; the tooling
deliberately does not silently invent acceptance criteria.

## Local tests (no CUDA required)

```bash
PYTHONPATH=poc python -m unittest discover -s poc/tests -v
```

## Provenance checklist

For each retained run, save outside Git:

- repository commit and `git diff --stat` (a dirty tree must be stated);
- `conda env export`, `pip freeze`, NVIDIA driver, CUDA toolkit, and GPU name;
- checkpoint kind, original filename/source, and SHA-256;
- IAM archive/preprocessing provenance and hashes for `labels.pkl` and the line list;
- split, start/IDs, thresholds, NMS, binarization threshold, and output hashes;
- manual visual-QA decision and any excluded lines, with reasons.
