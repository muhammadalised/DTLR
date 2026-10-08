# IAM matched text-only subword controls (2026-10-08)

The controls compare complete handwriting-aware vocabulary selection against
text-only selection, holding the IAM training text, 79 singleton characters,
145-piece vocabulary budget, normalization, and algorithm-family encoders fixed.
All five compared models have 225 CTC classes. The original frozen 222-class
bigram artifact is unchanged; its previously derived training-alphabet copy is
used in this descriptive comparison.

## Predeclared policy

Use only the existing pinned 5,694 IAM training transcriptions. Text controls
receive line IDs and text, with no edges, boxes, component IDs, or connectivity
counts. Their trainers work on dictionaries with no `edges` key. CCL evidence is
loaded after fitting, solely for the descriptive comparison. No raw images are
needed by the controls. The same dataset/split/hash/line-identity checks bind the
training corpus to the existing experiment.

- Case and original spaces/punctuation are preserved; normalization is NFC.
- Candidate pieces are letters only, length 2–6, with raw text count at least 20.
- Both controls retain all 79 singleton fallbacks and select 145 pieces.
- Text BPE ranks current adjacent-token pairs by their text occurrence count,
  recomputes after each merge, and uses the same ordered-merge encoder as HW BPE.
- Text Unigram seeds at most 580 pieces by text count, initializes with text
  counts, and uses the same exact EM, eight iterations per stage, individual
  fixed-weight removal-loss pruning, shrink factor 0.75, and Viterbi encoder as
  HW Unigram. This custom implementation is not SentencePiece.
- Freeze `predeclared_policy.json` before fitting. These inherited settings are
  exploratory, not optimized by the resulting comparison.

The count criterion deliberately differs: HW candidates need 20 exact/usable
image observations and connected rate at least 0.5, while text candidates need
20 raw textual occurrences. Consequently this is a comparison of the full HW
selection policy, including evidence availability. It does not isolate the
connectivity-rate filter alone. Unigram shares the seed cap and training policy,
not the initial candidate count: HW has 303 admitted pieces and text has 3,325.

The shared implementation exposes an explicit `handwriting_aware=False` control
mode. Text models use separate schema `dtlr.text-subwords.v1` and algorithm names
`text-only-bpe`/`text-only-unigram`; they contain no CCL connectivity fields or
invented image observations. Existing default HW behavior is preserved.

## Results: IAM train only

| Model | Tokens | Reduction vs characters | Connected edges cut | Observed whole-span support |
|---|---:|---:|---:|---:|
| HW bigram, derived alphabet | 181244 | 25.07% | 50.78% | 65.07% |
| HW BPE | 175612 | 27.40% | 45.92% | 61.89% |
| Text BPE | 155855 | 35.56% | 39.89% | 49.92% |
| HW Unigram | 177264 | 26.71% | 47.25% | 61.83% |
| Text Unigram | 155789 | 35.59% | 40.46% | 48.95% |

The same 241,876 characters and 78,957 observed connected edges underlie all
conditions. Span support excludes abstentions and divides by known emitted
multi-character spans. Numerators, denominators, and token-length histograms
are retained in the report. Text controls compress more and also cut fewer
connected edges. The HW models have higher observed whole-span support. These
are different tradeoffs; no overall model superiority is established.

At length 3, observed support is 58.25% for HW BPE versus 41.20% for text BPE,
and 57.35% for HW Unigram versus 41.91% for text Unigram. The HW support proportion
is higher at every shared emitted length, but the underlying token strings and
occurrences still differ. The same training connectivity evidence was used for
HW selection and these measurements, so the support difference is not an
independent validation of handwriting awareness. This is not controlled causal evidence or held-out
visual accuracy. Rare long pieces have especially small denominators.

BPE vocabularies share 59 of 145 pieces; Unigram vocabularies share 50.
The first HW BPE merges are `h+e`, `e+r`, `t+he`; the text control starts with
`t+h`, `i+n`, `th+e`. Both can learn `the` through different merge paths.
For illustrative text `the handwriting`, HW BPE emits
`the | SPACE | h | an | d | w | r | i | t | i | n | g`, while text BPE emits
`the | SPACE | h | and | w | rit | ing`.

## Verification and reproduction

All five encoders round-trip all IAM training lines without blank emission.
All prior HW comparison measurements reproduce exactly. Default HW training was
also rebuilt to verify that the control-mode extension preserves its learned
vocabulary, IDs, merge order/probabilities, evidence records, policies, and EM
training curve. The frozen bigram SHA-256 remains unchanged. A second full build with
`PYTHONHASHSEED=19` produces exactly the same learned models (excluding changed
worktree provenance), byte-identical candidate JSON and Markdown report, and
identical metrics/examples/vocabulary differences. The CPU test suite
has 65 passing tests, including control training without edges and invariance to
changed connectivity when `handwriting_aware=False`.

Successful results are in `outputs/iam-text-controls-v2/`:
`text-bpe.json`, `text-unigram.json`, `text_candidates.json`, `comparison.json`,
`comparison.md`, and `comparison.html`. Policy, source snapshots, input/output
hashes, and worktree provenance accompany them. Models remain experimental,
not frozen. Nothing was pushed, and no TVA/downstream dataset is accessed.

Current model file hashes, including their recorded working-tree provenance:

- Text BPE: `284267b5246550b5fe77d23dc622d6ed8519a152e25478a7ddf350e3b6fdcc8f`
- Text Unigram: `5655053410b69c7e5026349156678d04b48f66d3111031b583c1d90d18573cef`

From `/home/artellisys/DTLR`, choose a new output directory:

```bash
PYTHONPATH=poc conda run --no-capture-output -n dtlr-poc python \
  poc/scripts/build_iam_text_subword_controls.py \
  --iam-train-dir /home/artellisys/dtlr-output/iam-train-full \
  --hw-model-dir outputs/iam-handwriting-subwords-v1 \
  --output-dir outputs/iam-text-controls-repeat
```

Full-file hashes change if source/worktree provenance changes. Learned parameters
should be compared separately from that provenance when checking repeatability.
The builder pins the prior HW snapshot and validates its measurements, rather
than silently substituting newly fitted HW models.

## Remaining research work

The longer-span AI-assisted visual pilot is recorded separately in
`iam-long-span-qa-pilot.md`; independent human review remains pending. Before
recognition claims, predeclare an IAM evaluation policy and run matched
recognition experiments. Optional ablations can separate evidence availability,
connectivity eligibility, initialization, merge ranking, and token-length effects.
No held-out IAM images or recognition experiments are part of this run.
