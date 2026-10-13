# DTLR IAM evidence generation and export handoff

## Current scope (2026-10-10)

DTLR owns IAM character localization, transcription alignment, connected-component
labeling (CCL), connectivity evidence generation, and portable evidence export.
TVA owns tokenizer vocabulary construction: bigram selection, handwriting-aware
BPE merge learning, and Unigram probability fitting/pruning. DTLR does not select
new tokenizer vocabularies as part of its current evidence workflow.

The current branch is `codex/iam-handwriting-bpe-unigram`. The evidence-export
migration and distribution notebook are included alongside this handoff. The
preceding committed revision is `21eca13` (IAM overview notebook). Preserve
unrelated changes and inspect Git status before further work.
Work and writes in this chat stay within `/home/artellisys/DTLR`; the pinned IAM
train exports may be read as inputs. Do not modify TVA or access FAU, READ, or
other downstream datasets. Do not push without explicit user authorization.

Read [the migration note](poc/provenance/tokenizer-training-moved-to-tva.md) and
[the current POC responsibility section](poc/README.md) for the ownership change.
The tokenizer construction commands and experiment results below are historical.

## Evidence pipeline and consumer contract

1. Run DTLR on IAM training line images to obtain predicted character boxes.
2. Align ordered predictions to ground-truth transcription characters.
3. Apply CCL to the binarized images and identify pair-facing dominant components
   using `dominant-core-v3`.
4. Retain adjacent-character offsets, usable exact-alignment component IDs,
   abstentions, and aggregate pair observation counts/connectivity rates.
5. Export this evidence for TVA to construct its tokenizer vocabularies.

Usable exact evidence preserves both component IDs. Unusable or non-exact
observations retain their offsets with null component IDs: these are abstentions,
not disconnected observations. Component IDs are comparable only within the same
line image. Retaining identity, rather than only a connected/disconnected flag,
allows a consumer to check component consistency across longer spans.

The export includes all non-whitespace adjacent-pair observations and aggregate
score rows, including low-count and low-rate pairs. It applies no vocabulary
threshold, letters-only candidate filter, merge learning, or probability fitting.
The historical count-20/rate-0.5 policy belongs to tokenizer construction, not to
evidence export. Raw images and detector boxes are not included in the bundle.

## Portable IAM training evidence v1

Implementation: `poc/dtlr_poc/tokenizer_evidence.py`.
CLI: `poc/scripts/export_iam_tokenizer_evidence.py`.
Schema: `dtlr.iam-tokenizer-evidence.v1`.

The compressed `iam_connectivity_v1.json.gz` bundle contains:

- sole dataset/split: IAM train; connectivity method: `dominant-core-v3`;
- 5,694 training line IDs and NFC transcriptions;
- 160,536 non-whitespace adjacent-pair observations;
- 1,259 aggregate score rows;
- the original selection, evidence manifest, and five source SHA-256 hashes;
- an explicit contract for edge fields, abstentions, and component-ID scope.

Pinned export checksums:

- compressed bytes: 859,995;
- compressed SHA-256:
  `815304d2591bc187a2eea7dd76593499905c11645c8b9a146e7f1699c0e636fa`;
- uncompressed SHA-256:
  `1b44fddd0fa76dbb4ed14517f4361c522a4415139d16307b3468e14d122b8d0c`.

The exporter authenticates all five IAM inputs, validates line identity/text
against the selection, and reconstructs primary aggregate scores from occurrence
rows. It refuses changed source hashes and refuses to overwrite an existing
export with different bytes. A changed evidence release needs a new version.
It runs independently of TVA and does not train a recognition model or tokenizer.

## Current export command

Generate the bundle inside DTLR's ignored outputs directory:

```bash
cd /home/artellisys/DTLR
conda run --no-capture-output -n dtlr-poc python \
  poc/scripts/export_iam_tokenizer_evidence.py \
  --iam-train-dir /home/artellisys/dtlr-output/iam-train-full \
  --output /home/artellisys/DTLR/outputs/iam-tokenizer-evidence-v1/iam_connectivity_v1.json.gz
```

The exporter reads these pinned files under the IAM train input directory:
`selection.json`, `detections.jsonl`, and
`bigrams-dominant-core-v3/{manifest.json,bigram_scores.json,bigram_evidence.json}`.
Their expected hashes are recorded in `tokenizer_evidence.SOURCE_SHA256`.
TVA's consumer then uses the portable evidence to select/fit tokenizers; consumer
training and recognition evaluation belong to that repository.

## Historical interfaces and analysis dependencies

The frozen Bigram, BPE, and Unigram artifacts remain unchanged under
`poc/frozen/`. DTLR retains their historical inference and descriptive comparison
interfaces; frozen-model inference does not import TVA.

Historical Python training functions now delegate lazily through
`poc/dtlr_poc/_tva_training.py` to `tva.handwriting.training`. These compatibility
calls require an installed TVA package or its checkout (`TVA_ROOT` or the sibling
checkout); they are not independent DTLR training implementations.

The former `build_iam_bigram_tokenizer.py`, `build_iam_handwriting_subwords.py`,
and `build_iam_text_subword_controls.py` CLIs are retired and direct users to
TVA's `train_ed_tokenizers.ipynb`. Do not run their historical commands below
as current DTLR generation instructions.

`IAM_tokenizer_distributions.ipynb` and `build_iam_span_review.py` still call
`collect_candidates`, so rerunning them now requires TVA through the compatibility
bridge. Their saved executed outputs remain historical results. The IAM overview
notebook and the portable evidence exporter do not require tokenizer fitting.

## Independently checked DTLR verification (2026-10-10)

The review ran 3 evidence-export tests and 10 frozen-subword tests successfully.
It regenerated the full bundle inside
`outputs/iam-evidence-review-20261010/iam_connectivity_v1.json.gz`; both hashes,
byte size, and line/edge/score counts matched the release above. Frozen artifacts
retain their published checksums. `git diff --check` passed.

These standalone checks can be rerun without invoking TVA training:

```bash
PYTHONPATH=poc conda run --no-capture-output -n dtlr-poc python \
  -m unittest discover -s poc/tests -p test_tokenizer_evidence.py -v
PYTHONPATH=poc conda run --no-capture-output -n dtlr-poc python \
  -m unittest discover -s poc/tests -p test_subword_freeze.py -v
```

The full historical suite contains vocabulary-construction tests that now
invoke TVA. Its results and TVA-side reproduction results reported in the
migration note were not independently rerun in this DTLR-only review.
Training statistics and frozen settings do not establish held-out connectivity
accuracy, optimal thresholds, or CER/WER improvements.

## Historical tokenizer work before the ownership migration

The sections below preserve completed experiments and their provenance. Earlier
scope descriptions, training commands, test totals, and statements that nothing
was pushed describe their original stages; the current scope and commands above
supersede them. The historical commits through `21eca13` were subsequently pushed
to GitHub and GitLab. The subsequent migration and distribution-notebook
changes are included alongside this handoff.

## Historical frozen-bigram scope

This branch freezes the generic English handwriting-bigram artifact learned
only from the IAM training split. It does not modify TVA, create a combined
IAM-plus-READ tokenizer, or inspect any downstream dataset. DTLR owns the
handwriting-evidence vocabulary; consuming repositories own their downstream
task alphabets and segmentation policies.

The artifact is
`poc/frozen/iam-english-handwriting-bigram-v1.json`, versioned as
`iam-english-handwriting-bigram-v1`.

## Evidence and selection policy

- sole dataset and split: IAM train
- connectivity method: `dominant-core-v3`
- score: `exact_alignment_connected_rate`
- minimum exact-alignment observations: 20
- connected-rate threshold: 0.5
- pair policy: letters only
- selected handwriting bigrams: 145
- singleton characters: 76
- complete vocabulary: 222, with the empty CTC blank at ID 0
- deterministic ID order: blank, Unicode-sorted singletons, token-sorted
  bigrams
- Unicode normalization: NFC before segmentation

Each selected-bigram record preserves its source count, connected rate, and
utility. The existing maximum-total-utility non-overlapping dynamic program
remains DTLR's reference segmentation behavior. This segmentation algorithm
was not used to select the vocabulary and need not be adopted by a downstream
consumer.

## Reproducibility result

The existing provisional model was rebuilt from the pinned score file using
the original policy. The rebuilt JSON was byte-identical to the source model,
confirming exact reconstruction of its 145 bigrams, 76 singleton characters,
222 IDs, utilities, and counts.

Pinned hashes:

- scores: `45c86a9f703ebc3684563ec2a840e7c34241aa76e84d692be46cf11bfec27055`
- evidence manifest: `82cb95a14e71ab5b70c868ca1676b3f628552fabac395fbd30039fc7798bf190`
- existing model: `36db922860cbd501a6c5a1297dbf7d998abdbe1a337513acb00b44e74fbd9334`
- frozen generic artifact:
  `2fbb81479211454d8ad028d8af76eb26e76b75b80408a1a1e1f4efa15ab26a92`

Freeze mode rejects changed hashes, unexpected schemas, non-IAM datasets,
non-training splits, a different connectivity method, inconsistent score
counts, changed vocabulary counts, and any source model that cannot be exactly
reconstructed.

## Historical generation command (retired CLI)

```bash
cd /home/artellisys/DTLR
PYTHONPATH=poc conda run -n dtlr-poc python poc/scripts/build_iam_bigram_tokenizer.py \
  --scores /home/artellisys/dtlr-output/iam-train-full/bigrams-dominant-core-v3/bigram_scores.json \
  --manifest /home/artellisys/dtlr-output/iam-train-full/bigrams-dominant-core-v3/manifest.json \
  --source-model /home/artellisys/dtlr-output/iam-train-full/tokenizer-demo/model.json \
  --model-version iam-english-handwriting-bigram-v1 \
  --minimum-count 20 --rate-threshold 0.5 --pair-policy letters-only \
  --output poc/frozen/iam-english-handwriting-bigram-v1.json
```

Run the CPU-only test suite with:

```bash
PYTHONPATH=poc conda run -n dtlr-poc python -m unittest discover -s poc/tests -v
```

## Limitation and next consumer step

The count 20 and rate 0.5 thresholds were originally provisional
demonstration settings. Freezing them unchanged before downstream evaluation
makes the experiment reproducible and prevents outcome-driven tuning, but does
not scientifically establish them as optimal or held-out validated.

The next repository may consume the frozen 145-bigram inventory, define its
own task singleton alphabet, and explicitly declare how overlapping bigrams
are segmented. Any such adapter must be versioned and described as derived;
it is not the unchanged DTLR evidence artifact.

## Experimental handwriting-aware BPE / Unigram extension (2026-10-08)

Branch `codex/iam-handwriting-bpe-unigram`, based on the local frozen-bigram commit
d2f4631 of `dtlr-iam-bigram-poc`, adds an IAM-train-only variable-length tokenizer prototype.
Read `poc/provenance/iam-handwriting-subwords-poc.md` for the full method,
reproduction command, training settings, limitations, and next evaluation steps.
New implementation: `poc/dtlr_poc/handwriting_subwords.py`;
CLI: `poc/scripts/build_iam_handwriting_subwords.py`;
tests: `poc/tests/test_handwriting_subwords.py`.

Reuse existing occurrence-level dominant-core-v3 evidence. Longer candidates
require one consistent component across all usable/exact pair-facing cores;
this is an exploratory span extension requiring visual QA. BPE ranks current
merges by image-supported occurrence counts. Unigram constrains its candidate
vocabulary with CCL evidence, then fits a text unigram LM with exact EM and
individual fixed-weight removal-loss pruning. It is a custom implementation,
not SentencePiece and not a joint image/text likelihood.

The full IAM-train text has three isolated symbols absent from the frozen
76-singleton layer: `(`, `#`, and `&`. A separately labelled derived bigram
comparison copy appends them, preserving all original IDs, bigrams, utilities,
and counts. All three comparison models use 79 singletons, 145 subwords, and
blank ID 0 (225 CTC classes). The frozen 222-class artifact is unchanged.

Generated models, candidate evidence, hashes, and descriptive train comparison
are under `outputs/iam-handwriting-subwords-v1/` (ignored by Git). These outputs
are prototypes, not frozen artifacts or recognition-performance results.
No TVA, READ, FAU, or other downstream data is used, and nothing is pushed.

Verification: all 56 CPU tests passed; all three models round-trip every
IAM-train line without blank emission; two full builds with different Python
hash seeds regenerate models and reports byte for byte. Train-only reductions
versus characters are 25.07% (derived HW bigram), 27.40% (HW BPE), and 26.71%
(HW Unigram). See the provenance note for connectivity tradeoffs, denominators,
output hashes, and the limits of these descriptive measurements.

## IAM longer-span QA and matched text controls (2026-10-08)

The next three DTLR steps were carried out in order on
`codex/iam-handwriting-bpe-unigram`: an AI-assisted visual QA pilot, text-only
BPE/Unigram controls, and a matched descriptive IAM training comparison.

- Read `poc/provenance/iam-long-span-qa-pilot.md` for the 22-span stratified
  inspection. All 71 box-present pair checks and five missing-pair checks
  reproduce the saved evidence. Real examples expose empty cores and an
  internal localization failure; human annotations remain pending. The rule
  and existing models were not changed to fit this review.
- Read `poc/provenance/iam-text-subword-controls.md` for the declared control
  policy and complete results. Controls use only IAM training text and the
  same alphabet, vocabulary budget, and algorithm-family encoders. A separate
  schema identifies text models; training does not read CCL edges.
- The successful comparison and control models are under ignored
  `outputs/iam-text-controls-v2/`. All five models round-trip all 5,694 training
  lines without blank emission. Text controls compress more; HW models have
  higher observed component support, including within shared token lengths.
  These measurements do not establish held-out recognition gains.

The full CPU suite now has 65 passing tests. Default HW learned parameters and
prior comparison measurements were verified unchanged. The frozen bigram
artifact retains its pinned hash. Notebook work remains separate. No TVA,
READ, FAU, or other downstream dataset is accessed, and nothing is pushed.
Next scientific step: independent predeclared review and an IAM recognition
comparison/evaluation policy before training/evaluation claims.

## Frozen HW BPE / Unigram experimental releases (2026-10-08)

The audited HW models are now released under fixed versions:

- `poc/frozen/iam-english-handwriting-bpe-v1.json`
- `poc/frozen/iam-english-handwriting-unigram-v1.json`
- `poc/frozen/iam-english-handwriting-subwords-v1.manifest.json`

Status is `frozen-experiment-policy`. Both retain the exact original 79-singleton,
145-subword, 225-class IAM-training-only models; BPE merges, Unigram probabilities,
IDs, evidence counts, policies, and original training provenance are preserved.
The existing frozen 222-class bigram is byte-identical. Text-only controls remain
separate prototypes. Read `poc/provenance/iam-handwriting-subwords-v1-freeze.md`
for checksums, scientific limitations, reproduction, and loading examples.

New freeze implementation: `poc/dtlr_poc/subword_freeze.py`; CLI:
`poc/scripts/freeze_iam_handwriting_subwords.py`. The existing subword encoder
validates frozen metadata and exact source reconstruction. The strict loader
also checks actual file bytes. Generation pins source/candidate hashes and
refuses to overwrite changed release files. `--check` verifies without writing.

Verification: 75 CPU tests pass; every IAM train line encodes identically to the
original models with exact round trips and no blank; a second freeze build with
`PYTHONHASHSEED=19` is byte-identical. Frozen status fixes the experiment before
recognition evaluation and does not claim optimal thresholds, held-out visual
validation, or recognition gains. The user's qualitative gallery review is
recorded as informal feedback; per-case human annotations remain pending.
Nothing is pushed and no downstream dataset is accessed.

## IAM pair and token distribution notebook (2026-10-08)

`poc/notebooks/IAM_tokenizer_distributions.ipynb` provides a simple descriptive
comparison of the frozen HW models on the pinned IAM training corpus. It
separates raw overlapping adjacent-letter frequencies, vocabulary membership,
and actual emitted token frequencies. All models use their existing declared
encoders; the bigram comparison uses only the established in-memory singleton
adapter, preserving original IDs, pairs, utilities, and the frozen file.

The notebook reconstructs the pinned pair scores from occurrence evidence,
finds 861 distinct letter pairs / 153,350 occurrences, and reproduces the
303-piece eligible span pool. All 12 code cells execute successfully on all
5,694 lines; emitted token totals and supported-span counts match the existing
comparison exactly. Every line round-trips without blank emission, and input
and frozen-model hashes remain unchanged. Six plots were visually checked.

An executed copy and seven aggregate CSV tables are under ignored
`outputs/iam-tokenizer-distributions-v1/`. Supported, disconnected, and abstained
categories and their denominators are explicit. This analysis reads no raw
images or downstream datasets and does not establish recognition accuracy.
