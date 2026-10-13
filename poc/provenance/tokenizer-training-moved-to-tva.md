# Tokenizer training moved to TVA — 2026-10-10

The user requested a shared workflow: `.train()` should perform vocabulary
selection/fitting for both linguistic and handwriting-aware TVA tokenizers.
DTLR now owns character localization/alignment/CCL evidence and portable export;
TVA owns bigram selection, supported BPE merges, and Unigram EM/pruning.

The original selection algorithms and deterministic tie rules were moved into
TVA without policy changes. DTLR retains frozen inference and historical
analysis. Public historical trainer functions delegate lazily to TVA rather
than carrying duplicate algorithm implementations. Former training CLIs point
to TVA's notebook. No DTLR recognition model, image preprocessing, alignment,
CCL method, split definition, evidence file, or frozen tokenizer is changed.

The portable v1 export bundles all 5,694 pinned IAM train transcriptions,
160,536 non-whitespace adjacent-pair observations, and 1,259 aggregate score rows.
It stores unusable/non-exact observations as explicit abstentions and retains
within-image component identity for longer-span consistency checks.
The compressed bundle is 859,995 bytes; SHA-256:
`815304d2591bc187a2eea7dd76593499905c11645c8b9a146e7f1699c0e636fa`.
Its uncompressed payload SHA-256 is
`1b44fddd0fa76dbb4ed14517f4361c522a4415139d16307b3468e14d122b8d0c`.

The export script has no dependency on TVA and performs no candidate filtering,
merge learning, or probability fitting. Historical DTLR freeze metadata remains
historical; TVA records separate reports for its own fitting/reproduction checks.
At migration validation on 2026-10-10, these changes were not yet committed
or pushed; this note records that validation stage.

Validation: DTLR's full suite has 78 passing tests. A fresh process regenerated
byte-identical evidence without importing TVA. TVA's full suite ran 93 tests
(86 passed, 7 optional skips); all three full handwriting fits reproduced both
the native frozen files and ED projections exactly. The updated ED notebook
executed successfully and all prior artifact checksums were preserved.
