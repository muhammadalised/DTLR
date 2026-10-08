# IAM longer-span visual QA pilot (2026-10-08)

This is a train-only, AI-assisted diagnostic inspection of selected BPE/Unigram
pieces. It is not an independent human validation or a population accuracy estimate.
No vocabulary, merge, probability, CCL rule, or selection threshold was changed.

## Fixed sampling and mechanical verification

The union of selected pieces longer than two characters contains 68 tokens:
52 length-3, 13 length-4, one length-5 (`ather`), and two length-6 (`better`, `letter`).
A deterministic SHA-256 ranking, seed `iam-long-span-qa-pilot-20261008-v1`, selects
two occurrences per available length/status stratum, preferring distinct tokens
and lines. Status strata are supported, disconnected, and abstained. A separate
four-case interior-component-disagreement sample was requested; its pool was
empty among occurrences of these 68 selected tokens. This does not prove that
such disagreements cannot occur in other substrings. There were no length-5
abstentions in this pool.

The sample contains 22 spans from 22 IAM training images: eight supported, eight
disconnected, and six abstained. Membership in a learned vocabulary is distinct
from actual emission; the queue records both. In this sample, all occurrences
belonging to Unigram are emitted there, whereas several pieces are not BPE pieces.

Before opening images, freeze `sampling_queue.json`, SHA-256
`64d5026391a5a6ff5406bfed27f49c729e8133658aeab849ef47c8c5841953c3`.
All selection, detection, evidence, score, manifest, and frozen-model inputs must
match the SHA-256 digests recorded by the existing tokenizer snapshot. Reconstruct
the complete candidate inventory from the pinned exports before sampling.

Read only the exact IAM train images named in those detections. Verify dimensions,
record image SHA-256 hashes, and recompute full-line Otsu/8-connected CCL. All 71
sampled pair comparisons for which both detections exist reproduce component IDs,
pixel counts, core and full-box geometry, alignment operations, usability, and
connectivity. A repeat run also verifies all five sampled missing-detection pair rows and
reproduces the same 22 selected spans and byte-identical diagnostic cards.
Pairs with a missing detection remain abstentions. Perturbing Otsu
by -10/+10 changes none of these 22 span decisions; that is a small diagnostic,
not a tuned threshold or broad robustness result.

## Visual findings

All 22 diagnostic cards were inspected, including unmarked originals, boxes,
whole components, and individual pair-facing cores. Full-line context was also
inspected for ambiguous abstentions. Annotations are saved as AI-assisted notes
in `outputs/iam-long-span-qa-v1/ai_assisted_review.json`. Human annotations remain
pending in a separate template.

- Supported examples (`ces`, `men`, `been`, `ffer`, `ather`, `letter`, `better`)
  show visibly connected main strokes. The same component may extend beyond the
  selected substring; this rule establishes shared support, not a component
  containing exactly the token and nothing else.
- Disconnected examples show visible breaks within the span. For `been`, the
  final `n` is separate. The same string `better` appears fully connected in
  one sample, split in another, and abstained in two others. A learned text
  tokenizer can emit `better` for all these occurrences: handwriting influences
  vocabulary training, while inference is text-only.
- The six abstentions include three missing-character cases, one non-exact
  `r`/`g` alignment in `Cherbourg`, and two empty right-core cases. Exact predicted
  character labels do not establish correct localization. In `des` from
  `described`, the `s` box sits on the end of the preceding connected ink;
  visual character attribution is uncertain. In the abstained `the`, the full
  word is visibly joined but the exclusive `e` core falls beyond its ink.
- A useful failure appears inside the disconnected `letter` example
  `g06-026c-03:26:32`: the second `t` box/core is narrow and reaches into the
  following `e`. The `tt` edge is judged disconnected even though the two tall
  strokes visibly connect. The whole six-character span still has genuine
  breaks, so its whole-span disconnected outcome does not expose this edge
  error. Whole-span agreement must not be used to claim each internal edge
  was correct.

No obvious whole-span binary contradiction was found in this small inspection,
but this is not a validated accuracy result. The localization issue and empty
cores are concrete limitations worth reporting. Keep the existing method and
models unchanged; collect independent, predeclared review before making a
scientific validation claim.

## Reproduce and review

From `/home/artellisys/DTLR`:

```bash
PYTHONPATH=poc conda run --no-capture-output -n dtlr-poc python \
  poc/scripts/build_iam_span_review.py \
  --iam-train-dir /home/artellisys/dtlr-output/iam-train-full \
  --iam-image-dir /home/artellisys/dtlr-data/IAM_new/data/imgs/lines \
  --model-dir outputs/iam-handwriting-subwords-v1 \
  --output-dir outputs/iam-long-span-qa-repeat
```

The output directory must be new. The browser gallery is `index.html`, machine
checks are in `review_queue.json`, and `human_review_template.json` is separate
from AI-assisted notes. All generated crops remain ignored under DTLR/outputs.
No downstream images or labels are accessed, and nothing is pushed.
