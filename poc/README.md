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
  reintroduce a one-column overlap. Raster-safe `exclusive-core-v2.1` is also
  retained as rejected provenance because a component from one character can
  extend into both disjoint cores while a different component dominates the
  neighboring character. The primary `dominant-core-v3` method marks a pair
  `connected` only when the same CCL component has the unique largest pixel
  support in both raster-safe cores. Empty cores and tied largest components are
  unusable. This parameter-free candidate still requires validation on non-test
  data before it is treated as a handwriting boundary label. The observed
  failures and method changes are
  recorded in
  [`provenance/connectivity-method-change-2026-08-21.md`](provenance/connectivity-method-change-2026-08-21.md).
- Unusable pairs are abstentions, never disconnected labels. Evidence and QA
  exports include objective `unusable_reason_codes` for missing detections,
  inverted or collapsed core geometry, cores containing no ink, and tied
  dominant components. Visual causes such as partial-character or dot-only
  localization remain manual review annotations and are not inferred.
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
  --output-dir "${DTLR_OUTPUT_ROOT}/iam-test-small/bigrams-dominant-core-v3"

python poc/scripts/render_iam_qa.py \
  --detections "${DTLR_OUTPUT_ROOT}/iam-test-small/detections.jsonl" \
  --data-root "${DTLR_DATA_ROOT}" \
  --output-dir "${DTLR_OUTPUT_ROOT}/iam-test-small/qa-dominant-core-v3"
```

The second command writes row-level evidence and split-specific aggregates in
both CSV and JSON, plus a manifest. The third command writes one side-by-side
box/alignment and CCL overview per line, enlarged pair-level crops under
`qa/pairs/`, a browser-friendly `qa/index.html`, and a `qa_manifest.json`. In a
connected pair crop the component dominant in both exclusive cores is magenta.
For a disconnected pair, the component dominant in the left core is blue and
the component dominant in the right core is orange; unrelated ink is gray. The
rejected `core-v2.1`, `core-v2`, and `box-v1` results remain visible for
comparison. Before scaling up,
manually inspect the crops and record an acceptance decision. This
visualization/acceptance step remains an explicit milestone gate; the tooling
deliberately does not silently invent acceptance criteria.

## Frozen validation gate

After the test-only engineering smoke check, freeze validation IDs before
running inference or inspecting their images. The selection rule ranks all
validation IDs by SHA-256 of the declared seed and ID, so it is deterministic
without relying on a language-runtime random-number implementation. The
selection manifest also pins the `labels.pkl` and transcription hashes.

```bash
python poc/scripts/freeze_iam_selection.py \
  --data-root "${DTLR_DATA_ROOT}" \
  --split valid --count 32 --seed iam-dominant-core-v3-20260821 \
  --output "${DTLR_OUTPUT_ROOT}/iam-valid-32/selection.json"

python poc/scripts/prepare_iam_selection_images.py \
  --selection-manifest "${DTLR_OUTPUT_ROOT}/iam-valid-32/selection.json" \
  --raw-lines-root "/absolute/path/to/extracted/iam/lines" \
  --data-root "${DTLR_DATA_ROOT}" \
  --output-manifest "${DTLR_OUTPUT_ROOT}/iam-valid-32/preprocessing.json"

python poc/scripts/export_iam_detections.py \
  --data-root "${DTLR_DATA_ROOT}" \
  --checkpoint "${DTLR_WEIGHTS_ROOT}/finetuned/IAM/checkpoint.pth" \
  --checkpoint-kind iam-finetuned --split valid \
  --selection-manifest "${DTLR_OUTPUT_ROOT}/iam-valid-32/selection.json" \
  --threshold 0.3 --nms 0.5 \
  --output "${DTLR_OUTPUT_ROOT}/iam-valid-32/detections.jsonl"

python poc/scripts/build_bigram_evidence.py \
  --detections "${DTLR_OUTPUT_ROOT}/iam-valid-32/detections.jsonl" \
  --data-root "${DTLR_DATA_ROOT}" \
  --output-dir "${DTLR_OUTPUT_ROOT}/iam-valid-32/bigrams-dominant-core-v3"

python poc/scripts/render_iam_qa.py \
  --detections "${DTLR_OUTPUT_ROOT}/iam-valid-32/detections.jsonl" \
  --data-root "${DTLR_DATA_ROOT}" \
  --output-dir "${DTLR_OUTPUT_ROOT}/iam-valid-32/qa-dominant-core-v3"

python poc/scripts/build_qa_review_queue.py \
  --qa-manifest "${DTLR_OUTPUT_ROOT}/iam-valid-32/qa-dominant-core-v3/qa_manifest.json" \
  --agreement-audit-count 100 --seed iam-v3-agreement-audit-20260822 \
  --output-dir "${DTLR_OUTPUT_ROOT}/iam-valid-32/qa-dominant-core-v3/review"
```

The review queue contains every v2.1/v3 disagreement, every unusable pair, and
a deterministic hash-selected audit of ordinary agreements. Open
`review/review_queue.html`; progress is saved locally in the browser. Export the
completed `manual_review.json` from the page and retain it with the run outputs.
The JSON/CSV queue files pin the review scope independently of browser state.

Validate and summarize the exported annotations before accepting the review:

```bash
python poc/scripts/summarize_qa_review.py \
  --queue "${DTLR_OUTPUT_ROOT}/iam-valid-32/qa-dominant-core-v3/review/review_queue.json" \
  --manual-review "${DTLR_OUTPUT_ROOT}/iam-valid-32/qa-dominant-core-v3/review/manual_review.json" \
  --output "${DTLR_OUTPUT_ROOT}/iam-valid-32/qa-dominant-core-v3/review/manual_review_summary.json"
```

The command exits nonzero for pending fields or internally inconsistent labels.
It reports observed outcomes by queue stratum; it does not misrepresent the
stratified review queue as a simple random accuracy sample.

Inspect the frozen-review false disconnections and separately recorded ad-hoc
discoveries without changing the review or connectivity implementation:

```bash
python poc/scripts/inspect_pair_failures.py \
  --detections "${DTLR_OUTPUT_ROOT}/iam-valid-32/detections.jsonl" \
  --data-root "${DTLR_DATA_ROOT}" \
  --selection poc/config/iam_v3_failure_cases_20260822.json \
  --output-dir "${DTLR_OUTPUT_ROOT}/iam-valid-32/failure-diagnostics"
```

Open `failure-diagnostics/index.html`. Each case includes the original
grayscale crop, the Otsu binary mask, all 8-connected components, dominant and
shared core evidence, a threshold sweep, and per-component pixel shares in
`failure_diagnostics.json`. The two ad-hoc cases remain explicitly excluded
from the frozen 169-pair review summary.

After completing the frozen review, compare shared-component support across
all v2.1/v3 disagreements:

```bash
python poc/scripts/analyze_disagreement_support.py \
  --detections "${DTLR_OUTPUT_ROOT}/iam-valid-32/detections.jsonl" \
  --data-root "${DTLR_DATA_ROOT}" \
  --queue "${DTLR_OUTPUT_ROOT}/iam-valid-32/qa-dominant-core-v3/review/review_queue.json" \
  --manual-review "${DTLR_OUTPUT_ROOT}/iam-valid-32/qa-dominant-core-v3/review/manual_review.json" \
  --output-dir "${DTLR_OUTPUT_ROOT}/iam-valid-32/disagreement-support-analysis"
```

The bidirectional-support feature is the maximum, across components present in
both cores, of the smaller left/right core pixel share. The report contains
label-stratified distributions and exploratory threshold metrics. It does not
change v3, include ad-hoc discoveries, or present validation-selected metrics
as held-out performance.

If any selected processed line image is absent, prepare that exact line using
the same recorded IAM/PyLaia preprocessing procedure as the smoke run. Do not
replace a difficult or failed selected line with another line; record missing
or failed inputs as such.

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
