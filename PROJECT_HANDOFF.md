# DTLR IAM handwriting-bigram handoff

## Current scope

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

## Generation command

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
