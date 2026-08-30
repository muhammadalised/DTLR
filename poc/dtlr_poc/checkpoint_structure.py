"""Strict, inspectable classification of DTLR checkpoint module layouts."""

import re
from collections.abc import Iterable


DECODER_CLASS_EMBED_PREFIX = "transformer.decoder.class_embed"


def classify_decoder_class_embed(
    state_keys: Iterable[str], decoder_layers: int
) -> dict[str, object]:
    """Classify the decoder classifier registration saved in a state dictionary.

    Base DINO checkpoints register ``class_embed`` as a layer-indexed ModuleList.
    READ fine-tuning with ``--new_class_embedding`` replaces that decoder-only
    registration with one Linear layer.  Reject partial and ambiguous layouts so
    callers never need to relax strict checkpoint loading.
    """
    if decoder_layers <= 0:
        raise ValueError("decoder_layers must be positive")

    keys = set(state_keys)
    singular_keys = {
        f"{DECODER_CLASS_EMBED_PREFIX}.weight",
        f"{DECODER_CLASS_EMBED_PREFIX}.bias",
    }
    present_singular = singular_keys & keys

    indexed_pattern = re.compile(
        rf"^{re.escape(DECODER_CLASS_EMBED_PREFIX)}\.(\d+)\.(weight|bias)$"
    )
    present_indexed: dict[int, set[str]] = {}
    for key in keys:
        match = indexed_pattern.match(key)
        if match:
            present_indexed.setdefault(int(match.group(1)), set()).add(match.group(2))

    if present_singular and present_indexed:
        raise ValueError("checkpoint has both singular and indexed decoder class_embed keys")
    if present_singular and present_singular != singular_keys:
        raise ValueError("checkpoint has an incomplete singular decoder class_embed")

    if present_indexed:
        expected_indices = set(range(decoder_layers))
        actual_indices = set(present_indexed)
        if actual_indices != expected_indices:
            raise ValueError(
                "checkpoint decoder class_embed indices do not match the configured "
                f"decoder layers: expected {sorted(expected_indices)}, got {sorted(actual_indices)}"
            )
        incomplete = [
            index for index, names in sorted(present_indexed.items())
            if names != {"weight", "bias"}
        ]
        if incomplete:
            raise ValueError(
                f"checkpoint has incomplete indexed decoder class_embed layers: {incomplete}"
            )
        return {
            "decoder_class_embed": "layer-indexed-module-list",
            "decoder_layers": decoder_layers,
        }

    if present_singular == singular_keys:
        return {
            "decoder_class_embed": "single-linear",
            "decoder_layers": decoder_layers,
        }

    raise ValueError("checkpoint has no recognized decoder class_embed parameters")
