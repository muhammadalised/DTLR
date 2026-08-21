# Connectivity method change: box intersection v1 → exclusive core v2

## Status

`box-intersection-v1` is rejected as a primary boundary label. Its result is
retained in exports for comparison and provenance. `exclusive-core-v2` is the
new candidate method and still requires validation on non-test IAM data.

## Triggering observation

Manual QA of the eight-line IAM test smoke run found cases where a component
belonging visibly to one character entered the overlap between two DTLR boxes.
The v1 rule therefore reported a shared component even though the component did
not contain ink from both characters. Two clear examples were:

- `c04-116-03`, GT indices `[27:28]`, pair `bi`: only the `b` component was
  highlighted, but it entered the overlapping `i` box.
- `c04-110-00`, GT indices `[30:31]`, pair `nd`: only the `d` component was
  highlighted, but it entered the overlapping `n` box.

The same QA included plausible connected examples (`My`, `em`) and plausible
disconnected examples (`ig`, `om`). These test examples diagnose the failure
mode; they must not be used to tune thresholds.

## Versioned definitions

- `box-intersection-v1`: connected when one CCL component intersects both full
  boxes. This is vulnerable to overlap-induced false positives.
- `exclusive-core-v2`: remove the horizontal overlap from both adjacent boxes,
  then require one CCL component to intersect both remaining cores. A pair is
  unusable when either core contains no ink component.

Exports use `exclusive-core-v2` for the primary `usable`, `connected`, and
aggregate rate fields. Explicit v1 and v2 fields remain side by side so the
behavior change is auditable rather than silent.

## Next validation gate

Run and visually review a fixed IAM validation-split sample before scaling or
exporting evidence for TVA. Do not select CCL, detection, threshold, or NMS
settings using the IAM test split.
