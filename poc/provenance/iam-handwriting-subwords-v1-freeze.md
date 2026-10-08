# Frozen IAM handwriting-aware BPE and Unigram v1

Date: 2026-10-08. Branch: `codex/iam-handwriting-bpe-unigram`.
Status: frozen experimental configuration, not a recognition-performance claim.
This release preserves the already audited IAM-training-only models without
refitting, changing selection settings, or incorporating downstream data.

## Release files

- `poc/frozen/iam-english-handwriting-bpe-v1.json`
- `poc/frozen/iam-english-handwriting-unigram-v1.json`
- `poc/frozen/iam-english-handwriting-subwords-v1.manifest.json`

The two models each contain 79 singleton characters, 145 selected multi-character
pieces, and the empty CTC blank at ID 0: 225 contiguous classes. Their original
vocabulary, IDs, selected-piece counts/rates, training policies, provenance,
ordered BPE merges, Unigram log probabilities, and EM curve are unchanged.
NFC normalization, case, spaces, and punctuation are preserved. Encoding emits
no blank; characters outside the frozen alphabet raise an error.

The older frozen bigram remains byte-identical at 222 classes. Its singleton
alphabet contains 76 characters; the new subword models preserve their original
full IAM-train alphabet of 79, including isolated `(`, `#`, and `&`. The previously
labelled derived bigram comparison copy remains a separate artifact.
Text-only controls are not part of this handwriting release.

## Checksums

| File | SHA-256 |
|---|---|
| HW BPE v1 | `a504c2306200faf6eab98f3feef6f98b81e6e6c6c7f707cf31aa3399a660be47` |
| HW Unigram v1 | `af2bc8455afd796367737857f3a36f1deb45f07ec9783a4bbd7f15057f2b9476` |
| Release manifest | `e8b42352a286dde90d6956ae22e92a33d8a94fcf8d251d2ce1cfe224492b6528` |

Pinned original model hashes:

- BPE: `9dedba1087e8e011633251b22159f857fb2e80babc15c877f7ffb03d7a985aad`
- Unigram: `1ea98cc926fa3658dc6455a8d9e24ebb3a42b72e15848e9aca0bb7bfa00753c5`
- Candidate pool: `a0d7cd4569aa8feffe65a8816b41e6579ab937efab8d9cf4ab25513b1762aebc`

The manifest retains all six original IAM selection/detection/evidence/score/
manifest/bigram hashes. No source export was modified. Source models and the
303-candidate pool remain under ignored `outputs/iam-handwriting-subwords-v1/`.
The original training provenance is historical and intentionally retained,
including its pre-commit snapshot description. The freeze separately identifies
reviewed implementation commit `9cecdf3574c7b3193a743f367fa61f12c2839239`.

## What freezing establishes

Version names and content are locked before recognition evaluation. The same
source model is recoverable exactly from each frozen artifact: remove the new
`model_version`/`freeze` fields, restore the original status, and serialize with
the original canonical JSON convention. Its SHA-256 must equal the pinned
original model hash. This protects every learned and provenance field, rather
than only vocabulary membership.

The encoder validates frozen version/status/metadata, canonical artifact hashes,
and exact source reconstruction. `load_frozen_subword_model` additionally verifies
the actual file bytes against the published SHA-256. The generation CLI checks
pinned source file and candidate hashes and evidence records, validates the
existing bigram, and refuses to overwrite different release files. It can verify
an existing release without writing. No dataset files are required for inference.

## Scientific limits and review status

The count-20/rate-0.5 thresholds, length cap of 6, and training settings remain
exploratory. This freeze does not establish optimality, generalization, or
recognition improvements. Longer spans use the conservative pair-facing
component-consistency extension and retain localization/abstention limitations.
Unigram uses visual candidate eligibility and initialization, followed by a text
likelihood; it is not a joint image/text model or a SentencePiece implementation.

The 22-span AI-assisted training-image pilot was followed by the user's informal
qualitative feedback that decisions looked reasonable and abstained examples
contained anomalies. Individual human annotations remain pending. That feedback
is not represented as a completed case-level or held-out validation. See
`iam-long-span-qa-pilot.md` and `iam-text-subword-controls.md` for the visual and
text-control results and limitations. Training support measurements reuse the
selection evidence and are not independent validation.

## Reproduction and use

From `/home/artellisys/DTLR`:

```bash
PYTHONPATH=poc conda run --no-capture-output -n dtlr-poc python \
  poc/scripts/freeze_iam_handwriting_subwords.py --check

# Regenerate in a separate directory; fixed v1 metadata yields identical bytes.
PYTHONPATH=poc conda run --no-capture-output -n dtlr-poc python \
  poc/scripts/freeze_iam_handwriting_subwords.py \
  --source-dir outputs/iam-handwriting-subwords-v1 \
  --output-dir outputs/iam-subword-freeze-reproduced
```

The freeze uses the original pinned snapshots, not a new fit with changed
provenance. The self-contained released JSON files can be used directly:

```python
from dtlr_poc.subword_freeze import load_frozen_subword_model
from dtlr_poc.handwriting_subwords import HandwritingSubwordTokenizer

model = load_frozen_subword_model("poc/frozen/iam-english-handwriting-bpe-v1.json")
tokenizer = HandwritingSubwordTokenizer(model)
ids = tokenizer.encode("the handwriting")
assert tokenizer.decode(ids) == "the handwriting"
```

The Unigram path uses the same API. Any alphabet, ID, or segmentation changes must
be labelled as a separately versioned derived artifact, preserving the original
release. Future selection-policy changes require a new experiment/version.

## Verification

- 75 CPU tests pass, including altered IDs, evidence counts, split/dataset,
  policy, merge order, renormalized probabilities, version metadata, file hashes,
  exact reconstruction, manifest checksums, and all-piece round trips.
- Both releases encode all 5,694 pinned IAM train lines with identical IDs to
  their sources, decode exactly, and never emit blank. Token totals remain
  175,612 (BPE) and 177,264 (Unigram).
- The release regenerated byte for byte with `PYTHONHASHSEED=19`.
- Frozen bigram SHA-256 remains
  `2fbb81479211454d8ad028d8af76eb26e76b75b80408a1a1e1f4efa15ab26a92`.

Only DTLR files and permitted IAM train exports were used. No raw images were
needed for freezing, no TVA or downstream dataset was accessed, and nothing
was pushed. The release is available for a predeclared recognition comparison;
that evaluation is separate work.
