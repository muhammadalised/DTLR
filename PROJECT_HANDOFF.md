# DTLR handwriting-aware bigram project handoff

## Purpose and current state

This repository contains an additive proof-of-concept pipeline for learning
which adjacent character pairs tend to be physically connected in handwriting.
The intended downstream consumer is TVA, but this repository does not modify
TVA.

The current milestone is complete for offline English IAM and Early Modern
German READ 2016 training data:

1. use dataset transcription for character identity;
2. use a dataset-fine-tuned DTLR checkpoint for approximate character boxes;
3. align the transcription monotonically to the ordered detections;
4. binarize the handwriting and label 8-connected ink components;
5. decide whether each adjacent pair shares its dominant physical ink
   component;
6. export split-specific evidence and aggregate scores; and
7. build a provisional combined IAM + READ bigram tokenizer.

The combined model has been generated successfully. It has 494 vocabulary
entries: one CTC blank, 91 single characters, and 402 eligible handwriting
bigrams. The next research milestone is a controlled TVA experiment on the
OnHW-words500 labels and IMU handwriting signals.

## Scientific idea

This is not an ordinary text-frequency bigram tokenizer. A linguistic bigram
method asks how often two characters occur together in text. This method asks,
among usable and exactly aligned handwritten occurrences of a pair, how often
the two character regions are dominated by the same connected ink component.

For a pair `xy`, its current utility is:

```text
exact-alignment connected rate =
    usable exact-alignment occurrences classified connected
    ---------------------------------------------------------
    all usable exact-alignment occurrences
```

The minimum observation count is only a reliability filter. Frequency alone
does not make a pair eligible. A frequent but usually disconnected pair can be
rejected, while a sufficiently observed pair with a high physical-connectivity
rate can be retained.

This approach complements the earlier forced-alignment proposal:

- forced alignment estimates character boundaries from sequence/model timing;
- DTLR + connected-component labelling uses image localization and actual ink
  topology;
- both can be evaluated, and neither should be presented as already proven
  superior on OnHW.

## Source responsibilities

Each stage has a deliberately narrow responsibility:

- **IAM/READ ground truth:** character identity and character order.
- **DTLR:** approximate character localization, not final physical connection.
- **Otsu threshold:** converts each grayscale line into an ink/background mask
  using a per-line data-derived brightness cutoff.
- **8-connected-component labelling:** groups ink pixels touching horizontally,
  vertically, or diagonally into physical ink components.
- **Alignment:** maps each ground-truth character position to an ordered DTLR
  detection through deterministic Levenshtein alignment. It records matches,
  substitutions, insertions, and missing detections; it does not repair them
  silently.
- **Pair evidence:** records localization, alignment, connectivity, abstention
  reasons, comparison methods, and provenance for every within-token adjacent
  pair.
- **Aggregate scores:** summarize evidence by `dataset`, `split`, and `pair`.
  Train, validation, and test evidence must remain separate.
- **Tokenizer:** uses dynamic programming to choose the maximum-total-utility
  set of non-overlapping eligible bigrams. It consumes text labels; it does not
  run DTLR or inspect a new handwriting sample during tokenization.

## Frozen connectivity decision

The primary method is `dominant-core-v3`:

1. remove the horizontal overlap between adjacent DTLR boxes;
2. rasterize the two remaining cores inward so they cannot share a pixel
   column;
3. count connected-component pixels in each core; and
4. report connected only when the same component has the unique largest pixel
   support in both cores.

Empty cores, tied dominant components, missing detections, and invalid core
geometry are unusable. Unusable means an abstention, not disconnected.

Earlier methods remain in every export for auditability:

- `box-intersection-v1` was rejected because overlapping boxes caused false
  connections when ink from only one character entered both boxes.
- `exclusive-core-v2` removed geometric overlap, but fractional coordinates
  could still rasterize onto a shared pixel column.
- `exclusive-core-v2.1` fixed raster overlap, but the same character component
  could leak into both cores without representing both characters.
- `dominant-core-v3` adds the component-ownership requirement without a learned
  support threshold.

An exploratory bidirectional-support threshold recovered only two of four
connected IAM disagreement cases at its best validation cutoff. The positive
sample was tiny and the class distributions overlapped, so the threshold was
not adopted as a hidden override.

See:

- [connectivity method changes](poc/provenance/connectivity-method-change-2026-08-21.md)
- [frozen IAM decision](poc/provenance/connectivity-method-freeze-2026-08-23.md)

## Verified results

### IAM smoke run

- split: test, engineering smoke check only
- lines: 8
- adjacent-pair evidence rows: 388
- aggregate pair rows: 194
- purpose: verify IAM-fine-tuned localization, alignment, exports, and QA

The smoke examples exposed overlap-induced false positives and motivated the
versioned connectivity-method changes. They were diagnostic examples, not a
threshold-tuning set.

### IAM frozen validation review

- frozen validation lines: 32
- all adjacent pairs: 884
- usable pairs in the QA manifest: 871
- manually reviewed stratified queue: 169/169
- alignment failures in the corrected review: 16
- evaluable v3 assessments: 146
- observed correct v3 assessments: 142
- observed v3 errors: 4, all false-disconnected
- observed false-connected errors: 0

The four in-queue false disconnections were `li`, `it`, `ai`, and `cu` at the
pair IDs recorded in the freeze document. Two further cases (`su` and `ch`)
were discovered ad hoc and were explicitly excluded from the frozen review
result. Because the queue intentionally contains all disagreements and
abstentions plus a hash-selected agreement audit, 142/146 is not a population
accuracy estimate.

Frozen provenance:

- queue SHA-256:
  `2acd0be3121bca9c85880d4bb9fa80cae2c78b255b6f8a89c83309bd7d8a6522`
- corrected manual-review SHA-256:
  `84a09949b86193e052e6037541b5ac91fb6e2cfb18b9d73b57e6a846f084a32b`
- detections SHA-256:
  `db89e4ae20cd892fccac5f5558238b35de759d4e22aa8002ec282d5f3b1f3aaf`

### IAM full training export

- lines: 5,694
- evidence rows: 160,536
- aggregate pair-score rows: 1,259
- v2.1/v3 disagreements: 11,036
- dominant-core-v3 unusable pairs: 1,683

These figures come only from the IAM training split and are suitable for model
construction. Dataset images, detections, reviews, and score exports remain
outside Git.

### READ validation transfer gate

- frozen validation lines: 32
- evidence rows: 449
- aggregate pair-score rows: 202
- v2.1/v3 disagreements: 25
- dominant-core-v3 unusable pairs: 16

Manual inspection was completed. It found generally good transfer, a small
number of exceptions, and plausible abstentions. One inconsistent annotation
was corrected before the review was accepted. Do not quote a READ review
accuracy until the corresponding review summary and its hashes are archived
with the external run outputs; those totals are not committed in this
repository.

### READ full training export

- lines: 8,367
- evidence rows: 133,432
- aggregate pair-score rows: 1,136
- v2.1/v3 disagreements: 11,113
- dominant-core-v3 unusable pairs: 5,832

The high READ abstention count is visible rather than converted into negative
labels. Common recorded causes include missing detections, empty cores, and
inverted cores.

### Combined IAM + READ tokenizer v1

- model version: `iam-read-combined-v1`
- status: provisional combined policy
- vocabulary size: 494
- eligible bigrams: 402
- inferred single-character entries: 91
- normalization policy: NFC
- default eligibility policy: at least 20 exact-alignment observations,
  connected rate at least 0.5, and two Unicode letters

Example audited model entries:

| Pair | Exact observations | Connected | Rate | Dataset detail |
| --- | ---: | ---: | ---: | --- |
| `he` | 6,254 | 5,038 | 0.8056 | IAM 0.7520; READ 0.9772 |
| `er` | 8,826 | 7,849 | 0.8893 | IAM 0.8046; READ 0.9385 |
| `ür` | 36 | 34 | 0.9444 | READ only |
| `ör` | 44 | 40 | 0.9091 | READ only |

Dataset-specific counts remain in the model so pooling is auditable. The
thresholds define a demonstration vocabulary policy; they are not a held-out
claim of optimality.

## Checkpoints and environment

Do not confuse language pretraining with dataset fine-tuning:

- `english-pretrained`: synthetic English initialization for IAM fine-tuning;
- `iam-finetuned`: checkpoint after IAM fine-tuning, used for IAM localization;
- `german-pretrained`: synthetic German initialization for READ fine-tuning;
- `read-finetuned`: checkpoint after READ fine-tuning, used for READ
  localization.

The supported execution host is Linux/WSL2 on the RTX 4060. Upstream DTLR uses
Python 3.11, PyTorch 2.1, CUDA 11.8, and custom CUDA operators. Do not attempt
the CUDA build on Apple Silicon. The primary environment is Conda, not Docker.

Keep the checkout and file-heavy datasets under the WSL filesystem, for
example `/home/artellisys/...`, rather than `/mnt/c/...`.

```bash
cd /home/artellisys/DTLR
conda activate dtlr-poc
source environment/conda/activate.sh
set -a
source .env
set +a
python environment/cuda-linux/preflight.py --require-gpu
```

Expected external layout:

```text
/home/artellisys/dtlr-data/
  IAM_new/
  READ_2016/
  read-raw-2016/PublicData/
/home/artellisys/dtlr-weights/
  finetuned/IAM/checkpoint.pth
  finetuned/READ/checkpoint.pth
/home/artellisys/dtlr-output/
```

Datasets, weights, Conda environments, native extension builds, detections, QA
pages, and large exports must not be committed.

## Scripts in creation order

This order follows repository history and is useful for reviewing how the
method evolved.

| Order | Commit | Script/module | Role |
| ---: | --- | --- | --- |
| 1 | `61d6e04` | `poc/dtlr_poc/alignment.py` | Monotonic GT-to-detection alignment. |
| 2 | `61d6e04` | `poc/dtlr_poc/ccl.py` | Otsu thresholding, 8-connected labels, and pair component evidence. |
| 3 | `61d6e04` | `poc/dtlr_poc/evidence.py` | Pair-row construction, aggregation, and CSV/JSON export. |
| 4 | `61d6e04` | `poc/scripts/export_iam_detections.py` | Run IAM-fine-tuned DTLR and write provenance-rich detections. |
| 5 | `61d6e04` | `poc/scripts/build_bigram_evidence.py` | Build pair evidence and split-specific scores. |
| 6 | `be7f666` | `poc/scripts/render_iam_qa.py` | Render line and pair-level visual QA. |
| 7 | `7a06992` | `poc/scripts/freeze_iam_selection.py` | Freeze deterministic IAM selections before inference. |
| 8 | `725a0a8` | `poc/scripts/prepare_iam_selection_images.py` | Reproduce selected IAM processed line images. |
| 9 | `023bba0` | `poc/scripts/build_qa_review_queue.py` | Freeze disagreements, abstentions, and an agreement audit for review. |
| 10 | `334abe0` | `poc/scripts/summarize_qa_review.py` | Validate completeness/consistency and summarize manual review. |
| 11 | `444609d` | `poc/scripts/inspect_pair_failures.py` | Render detailed diagnostics for known failure cases. |
| 12 | `5e5cc48` | `poc/scripts/analyze_disagreement_support.py` | Explore support distributions without changing v3. |
| 13 | `bbe2cda` | `poc/dtlr_poc/tokenizer.py` | Build/load models and encode/decode with dynamic programming. |
| 14 | `bbe2cda` | `poc/scripts/build_iam_bigram_tokenizer.py` | Build the provisional IAM-only model. |
| 15 | `bbe2cda` | `poc/scripts/tokenize_iam_bigrams.py` | Tokenization demonstration CLI; also loads combined models. |
| 16 | `c1a37bf` | `poc/dtlr_poc/read_dataset.py` | Resolve and validate READ examples and transcription normalization. |
| 17 | `c1a37bf` | `poc/scripts/freeze_read_selection.py` | Freeze READ selections. |
| 18 | `c1a37bf` | `poc/scripts/export_read_detections.py` | Run READ-fine-tuned localization with strict checkpoint reconstruction. |
| 19 | `c1a37bf` | `poc/scripts/render_read_qa.py` | READ-labelled wrapper around the frozen QA behavior. |
| 20 | `b71637b` | `poc/scripts/prepare_read_selection_images.py` | Reproduce PAGE-XML line crops and validate them against `labels.pkl`. |
| 21 | `6cde799` | `poc/scripts/build_combined_bigram_tokenizer.py` | Pool IAM/READ training counts and build combined v1. |
| 22 | `ae18fbf` / `3e1fff7` | `IAM_bigram_walkthrough.ipynb` | Presentation walkthrough, later updated for the combined model. |

Important behavior changes between these creations are preserved in commits
`f139aba`, `f1e7b6b`, and `df95c51`; these respectively introduced exclusive
cores, fixed fractional raster overlap, and introduced dominant-core-v3.

## Reproducing the completed pipeline

Full commands and safeguards are maintained in [the POC guide](poc/README.md).
The essential order for either dataset is:

```text
freeze selection
    -> prepare selected images
    -> export dataset-fine-tuned DTLR detections
    -> build bigram evidence and scores
    -> render QA
    -> freeze and complete manual review
    -> summarize review
```

Only after validation of the frozen method:

```text
freeze full training split
    -> prepare images
    -> export resumable detections
    -> build training-only scores
    -> build tokenizer model
```

Useful commands after cloning on a new WSL host:

```bash
git status --short --branch
git log --oneline -20
PYTHONPATH=poc python -m unittest discover -s poc/tests -v
jupyter lab IAM_bigram_walkthrough.ipynb
```

## Applying the tokenizer to TVA and OnHW-words500

OnHW-words500 contains IMU pen trajectories rather than offline page images.
The combined model therefore transfers a learned **label segmentation prior**,
not DTLR boxes or image connected components, into TVA.

The clean experiment is:

1. freeze the combined IAM + READ model and record its file hash;
2. inventory OnHW training-label Unicode coverage under NFC, including German
   characters and punctuation;
3. load the model through `HandwritingBigramTokenizer` and encode the OnHW word
   labels using its `vocab`, `idx_token`, `size`, `encode`, and `decode`
   interface;
4. configure the TVA output vocabulary/CTC head to that tokenizer size;
5. train TVA on the unchanged OnHW training signals with the encoded labels;
6. compare it with TVA's existing character and linguistic-bigram tokenizers
   under identical data splits, model capacity, seeds, and training budget;
7. report CER, WER, sequence length/compression, token coverage, and failures;
8. make policy choices only on training/development data and evaluate once on
   the untouched OnHW test split.

The comparison tests whether offline visual connectivity learned from IAM and
READ is a useful inductive bias for online IMU recognition. It does not assume
that offline ink topology and IMU stroke dynamics are identical. IAM Online
could later support a closer online-handwriting bridge, but adding it would be
a separate versioned experiment rather than an unreported change to combined
v1.

## Limitations that must remain visible

- DTLR boxes are approximate; alignment/localization failures contaminate pair
  evidence if not excluded.
- Dots and other detached diacritics naturally create difficult or empty-core
  cases.
- `dominant-core-v3` is conservative and has observed false disconnections.
- READ historical handwriting differs from contemporary OnHW German writing.
- The combined vocabulary thresholds are provisional and not yet optimized on
  held-out downstream performance.
- Manual QA queues are stratified diagnostics, not random population samples.
- A tokenizer score describes training-corpus evidence, not certainty for each
  future written instance.
- No claim about TVA recognition improvement exists until the controlled OnHW
  experiment is run.

## Git and artifact provenance

At the time this handoff was prepared, the active development line was
`dtlr-read-bigram-poc`, based through commit `3e1fff7` (`Update meeting notebook
for combined tokenizer`). Use `git log` to capture the newer handoff commit once
this file is committed.

For every retained experiment, archive outside Git:

- repository commit and dirty/clean status;
- Conda export and `pip freeze`;
- GPU, driver, CUDA toolkit, and PyTorch versions;
- checkpoint kind, original filename/source, and SHA-256;
- dataset/annotation hashes and frozen selection manifest;
- inference threshold, NMS value, split, and output hashes;
- review queue, manual annotations, summary, and their hashes.

