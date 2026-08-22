# IAM POC connectivity-method freeze: dominant core v3

## Decision

`dominant-core-v3` is frozen as the primary physical-connectivity label for the
IAM bigram proof of concept. `box-intersection-v1`, `exclusive-core-v2`, and
`exclusive-core-v2.1` remain comparison fields only. Bidirectional shared-core
support remains an exploratory diagnostic and must not override v3 in retained
POC exports.

This freeze applies to the IAM bigram POC, not to a general claim of optimality
for IAM, READ, other scripts, or downstream TVA performance.

## Frozen validation provenance

- Split: IAM `valid`
- Lines: 32, selected before inference with seed
  `iam-dominant-core-v3-20260821`
- Review queue: 169 pairs
- Queue SHA-256:
  `2acd0be3121bca9c85880d4bb9fa80cae2c78b255b6f8a89c83309bd7d8a6522`
- Corrected manual-review SHA-256:
  `84a09949b86193e052e6037541b5ac91fb6e2cfb18b9d73b57e6a846f084a32b`
- Detection SHA-256:
  `db89e4ae20cd892fccac5f5558238b35de759d4e22aa8002ec282d5f3b1f3aaf`

The original review export is retained outside Git. A corrected copy changed
`n06-194-07:15:16` (`'l`) from alignment-correct to alignment-incorrect after
the pair crop showed that the apostrophe box enclosed a stroke of the following
`l`. The correction was not made in place, so both review hashes remain
auditable.

## Corrected review result

- Completed review rows: 169/169
- Alignment failures: 16
- Evaluable v3 pairs: 146
- Correct: 142
- Errors: 4
- False disconnected: 4
- False connected: 0

The four observed in-queue errors were:

- `f07-032b-06:14:15` (`li`)
- `f07-032b-06:15:16` (`it`)
- `n06-194-07:1:2` (`ai`)
- `p03-096-02:5:6` (`cu`)

These are diagnostic outcomes from a stratified queue, not a population IAM
accuracy estimate. Two additional visual findings, `n03-079-05:8:9` (`su`) and
`n06-194-07:22:23` (`ch`), were discovered outside the frozen queue and remain
explicitly labeled ad hoc.

## Rejected support override

For all 56 v2.1/v3 disagreements, 48 pairs remained evaluable after the review
correction: four visually connected and 44 visually disconnected. The
exploratory score was the maximum, across shared components, of the smaller
left/right exclusive-core pixel share.

At the best exploratory validation cutoff, `0.32265446224256294`, the result
was two true connected, zero false connected, 44 true disconnected, and two
false disconnected. Precision was 1.0 and recall was 0.5, but only four
positive examples were evaluable and the connected/disconnected score
distributions overlapped. The cutoff therefore remains unfrozen and cannot be
reported as held-out performance.

The low-support `it` and `cu` cases show the cost of v3's conservative rule.
The shared components contributed only about 1.13% and 0.64%, respectively, to
one core. Accepting such asymmetric support globally would recreate the
component-leakage problem that motivated v3.

## Downstream contract

- Unusable pairs remain abstentions, never disconnected labels.
- Training, validation, and test aggregates remain separate.
- IAM training evidence may now be generated with this frozen method.
- Validation may guide tokenizer policy, but IAM test evidence must remain
  untouched until the tokenizer policy is frozen.
- Punctuation handling belongs to the tokenizer policy; it must not be hidden
  inside the physical-connectivity method.
