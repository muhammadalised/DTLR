# IAM handwriting-aware BPE and Unigram prototype

Status: original experimental prototype snapshot. Its exact learned HW BPE and
Unigram parameters were subsequently frozen as v1; see
[the freeze provenance note](iam-handwriting-subwords-v1-freeze.md). No held-out
validation or recognition gains are established. The ignored outputs listed
below retain their original prototype status and historical provenance.
The existing `iam-english-handwriting-bigram-v1` artifact remains unchanged.
All new models and measurements use IAM train only. No TVA files, READ evidence,
FAU data, or other downstream datasets are used.

## Research question

Can image-derived connectivity support variable-length subword vocabularies,
and how do their segmentations compare with the frozen handwriting bigrams?
The recognizer and CCL are not retrained. This experiment reuses the complete
IAM-train detections, transcriptions, and occurrence-level v5 CCL export.
Aggregate bigram rates alone would not identify whether a particular longer
span shares one component; the occurrence offsets and component IDs are essential.

## Longer-span evidence

For a letter-only substring of length 2 through 6, collect the adjacent-pair
rows from the same line. Every edge must have exact character alignment and
usable dominant-core-v3 evidence. Missing detections, non-exact alignments,
empty/tied cores, and missing edges cause abstention, not disconnection.
The substring is supported only when all left/right dominant component IDs
across those rows agree on one component. Otherwise it is an observed
unsupported span. Component IDs are compared only within an image.

This method is versioned as `pairwise-consistent-dominant-component-v1`.
It is a conservative extension using existing pair-facing cores, not a newly
validated multi-character detector. An interior character has two potentially
different pair-facing cores. The extra consistency requirement and longer
span observations need visual QA before scientific validation claims.

Candidate statistics retain total occurrences, exact/usable span observations,
connected occurrence counts, and their rate. Admit candidates with at least
20 exact/usable observations and a rate of at least 0.5. These cutoffs are
inherited exploratory settings, not statistically optimal thresholds. The
length cap of 6 is also an exploratory engineering choice. Tokens never merge
across spaces or punctuation; those characters remain singleton tokens.

## Handwriting-aware BPE

Start each training line as singleton characters. Among adjacent current token
pairs whose concatenation is an eligible candidate, rank merges by the number
of their current occurrences supported by one CCL component. Recompute that
support after each merge. Break ties deterministically by token text and pair.
Apply the chosen pair globally, left to right, including occurrences without
support: the output is a reusable vocabulary and merge list, not an
image-conditioned label sequence. Stop at 145 distinct subwords or when no
supported eligible merge is available. Inference follows the learned merge
order and requires only text.

This differs from ordinary BPE frequency ranking because the ranking count is
image-supported occurrence frequency. The vocabulary and merge order can
change when the connectivity evidence changes while the text stays fixed.
The underlying merge approach follows [Sennrich et al. (2016)](https://aclanthology.org/P16-1162/);
the visual constraint and merge priority are experimental DTLR choices.

## Handwriting-aware Unigram

Use the same CCL-admitted candidates. Seed at most four times the 145-subword
budget, ranked by connected occurrence counts, and initialize those pieces
from their connected counts. Mandatory singleton characters are always retained.
Fit independent token probabilities by exact forward-backward EM over all
text segmentations, with eight iterations per vocabulary stage. The likelihood
factorizes at forced singleton non-letter boundaries; this introduces no word
boundary or whitespace markers and preserves the original text.

For each removable piece, measure the training log-likelihood decrease when
its arcs are removed, holding current token weights fixed. This individual
removal loss is calculated exactly. Retain the pieces with largest loss and
shrink the subword inventory by 0.75 per stage until it reaches 145. Refit and
normalize probabilities with EM after each pruning stage. Individual loss
calculations omit interactions between pieces removed in the same batch.
Inference uses maximum-log-probability Viterbi segmentation.

This is a custom CCL-constrained unigram LM, based on the probabilistic
segmentation framework of [Kudo (2018)](https://aclanthology.org/P18-1007/).
It is not a SentencePiece implementation. Handwriting awareness enters its
candidate vocabulary and initialization; its EM likelihood is a text
likelihood, not an image likelihood or a connectivity penalty. A joint
visual/text objective is a separate future research variant.

## Shared alphabet and frozen reference

The full IAM-train text contains 79 distinct characters. The frozen artifact's
pair-derived singleton layer contains 76 and omits three symbols that occur
as isolated tokens: `(` (54 occurrences), `#` (47), and `&` (32).
The comparison uses an explicitly derived bigram copy that appends these
three singleton IDs while preserving every original ID, bigram, utility, and
count. It is labelled `experimental-derived-training-alphabet` and does not
claim to be the unchanged frozen artifact.

BPE and Unigram use the same 79-character alphabet. With blank ID 0 and 145
subwords, each comparison model has 225 CTC classes. All encoders preserve NFC,
case, spaces, and punctuation and emit no CTC blank. Unknown characters raise
an error. No unknown-token or byte fallback is silently introduced.

## Input checks and reproducibility

Before fitting, validate pinned manifest, score, and frozen-model hashes;
IAM/train/method/schema identity; all 5,694 selected line IDs and transcription
hashes in frozen order; unique and complete non-whitespace edge rows; matching
text offsets; and usable dominant-component consistency. Reconstruct primary
bigram totals, exact observation counts, and rates from the occurrence export
and match them against every pinned score. Record SHA-256 digests of the
selection, detections, evidence, relevant implementation files, Python version,
and repository commit. No input export is modified.

From `/home/artellisys/DTLR`, in the existing `dtlr-poc` environment:

```bash
PYTHONPATH=poc conda run --no-capture-output -n dtlr-poc python \
  poc/scripts/build_iam_handwriting_subwords.py \
  --iam-train-dir /home/artellisys/dtlr-output/iam-train-full \
  --max-token-length 6 --minimum-count 20 --rate-threshold 0.5 \
  --subword-budget 145 --em-iterations 8 \
  --output-dir outputs/iam-handwriting-subwords-v1

PYTHONPATH=poc conda run -n dtlr-poc python -m unittest discover -s poc/tests -v
```

Generated files remain under ignored `DTLR/outputs` to keep work within the
requested workspace. Models contain retained candidate evidence and all
training settings. `comparison.json` retains metric numerators/denominators,
token-length histograms, the frozen-alphabet audit, hashes, and illustrative
segmentations. `comparison.md` is the readable report. Synthetic tests check
that connectivity can change learned vocabularies, ordered merges can create
long tokens, forward-backward/Viterbi agree with exhaustive calculations,
removal losses match direct likelihood differences, and EM increases likelihood
within fixed-vocabulary stages. They also check abstentions, leakage guards,
CTC behavior, NFC, determinism, and preservation of the frozen reference.

## Interpretation and next research steps

The first comparison uses all IAM train lines and is descriptive. It measures
sequence reduction, connected-edge cuts, and the fraction of emitted
multi-character spans sharing one observed component. Connectivity metrics
exclude unusable/non-exact observations and expose their known denominators.
A tokenizer can cut fewer connected edges simply by emitting longer pieces;
that alone does not establish better units. Span support can decrease at the
same time, and different token lengths make raw span proportions difficult to
compare. Both measurements must be shown along with length histograms.

The experiment establishes an executable visual-evidence-to-subword pipeline.
It does not establish recognition gains, generalization, calibrated span
connectivity, or superiority of the handwriting signal over ordinary text
training. Next: visually inspect selected longer spans, predeclare validation
and evaluation policy, compare with text-only controls and length-restricted
ablations, and later measure CER/WER under matched recognition training.
Do not use these IAM-train descriptive results to claim held-out accuracy.

## Verified run snapshot (2026-10-08)

The complete 5,694-line IAM training corpus contains 241,876 characters.
303 candidates pass the declared CCL policy: 145 length-2, 125 length-3,
26 length-4, 5 length-5, and 2 length-6 pieces. Each trained comparison model
has 79 singleton characters, 145 multi-character pieces, and blank ID 0.

| Model | CTC classes | Tokens | Reduction vs characters | Connected edges cut | Emitted multi-token spans sharing one component |
|---|---:|---:|---:|---:|---:|
| Derived handwriting bigram | 225 | 181244 | 25.07% | 50.78% | 65.07% |
| Handwriting-aware BPE | 225 | 175612 | 27.40% | 45.92% | 61.89% |
| CCL-constrained Unigram | 225 | 177264 | 26.71% | 47.25% | 61.83% |

Each model passed encode/decode round trips on all training lines without
emitting blank. BPE emits tokens up to length 4 in this corpus; Unigram emits
up to length 6. BPE and Unigram shorten sequences and cut fewer observed
connected edges, but their emitted spans have lower whole-component support.
These are training-set tradeoffs, not a recognition-performance ranking.

All 56 CPU tests passed. Two full builds, including a run with
`PYTHONHASHSEED=19`, produced byte-identical models, candidate inventory,
comparison JSON, and Markdown report. The frozen artifact hash still matches
`2fbb81479211454d8ad028d8af76eb26e76b75b80408a1a1e1f4efa15ab26a92`.

Output SHA-256 digests for this implementation snapshot:

- Derived bigram: `06da23d5b33e743e66b943385eb1930146ed47843ffaaf16ded75980698411ac`
- HW BPE: `9dedba1087e8e011633251b22159f857fb2e80babc15c877f7ffb03d7a985aad`
- HW Unigram: `1ea98cc926fa3658dc6455a8d9e24ebb3a42b72e15848e9aca0bb7bfa00753c5`
- Candidate inventory: `a0d7cd4569aa8feffe65a8816b41e6579ab937efab8d9cf4ab25513b1762aebc`
- Comparison JSON: `c7db4f3384e9e929cdf73d7379537f8df0b3f2511477dd8679180d9d8b1882ae`

The JSON outputs contain the exact source and implementation hashes. These
hashes identify this uncommitted prototype snapshot; changing its implementation
or provenance fields requires recording a new result snapshot.

Exact pre-commit implementation bytes recorded by the experiment hashes are
retained under ignored `outputs/iam-handwriting-subwords-v1/source-snapshot/`.
The committed source normalizes line endings without changing the algorithms.
