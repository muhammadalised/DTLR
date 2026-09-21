# IAM English handwriting-bigram v1 provenance

## Decision

`iam-english-handwriting-bigram-v1` is the generic IAM-only artifact for the
next controlled experiment. It contains no combined-dataset evidence and no
downstream task data. Vocabulary selection and downstream label segmentation
are separate decisions:

- DTLR selects handwriting bigrams solely from IAM training evidence.
- The committed tokenizer retains maximum-total-utility non-overlapping DP as
  its reference segmentation implementation.
- A consumer may adapt the frozen 145-bigram list to its task alphabet and a
  separately declared segmentation policy. Such an adapter is not this frozen
  evidence artifact.

## Frozen evidence and policy

- Artifact schema: `dtlr.handwriting-bigram-tokenizer.v1`
- Model version: `iam-english-handwriting-bigram-v1`
- Evidence: IAM train only
- Connectivity method: `dominant-core-v3`
- Score: exact-alignment connected rate
- Minimum exact-alignment observations: 20
- Connected-rate threshold: 0.5
- Pair policy: two Unicode letters
- Unicode normalization before segmentation: NFC
- Ordering: blank, Unicode-sorted singletons, token-sorted bigrams
- CTC blank: empty string at ID 0
- Selected bigrams: 145
- Singleton characters: 76
- Complete vocabulary: 222

Every selected vocabulary row retains its exact observation count,
connectivity rate, and utility from the source model. IDs are contiguous from
0 through 221.

## Hashes and reconstruction audit

- IAM score file SHA-256:
  `45c86a9f703ebc3684563ec2a840e7c34241aa76e84d692be46cf11bfec27055`
- IAM evidence manifest SHA-256:
  `82cb95a14e71ab5b70c868ca1676b3f628552fabac395fbd30039fc7798bf190`
- Existing provisional model SHA-256:
  `36db922860cbd501a6c5a1297dbf7d998abdbe1a337513acb00b44e74fbd9334`
- Frozen generic artifact SHA-256:
  `2fbb81479211454d8ad028d8af76eb26e76b75b80408a1a1e1f4efa15ab26a92`

Re-running the legacy builder with count 20, rate 0.5, letters-only pairs, and
the required space singleton produced a byte-identical copy of the existing
model. Freeze mode additionally compares the reconstructed JSON object with
that pinned source model before writing the generic artifact.

## Exact command

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

## Scientific limitation

The 20-observation and 0.5-rate thresholds originated as transparent
demonstration settings, not as a held-out statistically optimized policy.
They are frozen unchanged here to make a pre-downstream experiment
reproducible and leakage-resistant. The artifact must not be described as
evidence that those cutoffs are optimal. Changing them would require a new,
independently justified and versioned policy.
