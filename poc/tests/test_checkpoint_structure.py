import unittest

from dtlr_poc.checkpoint_structure import classify_decoder_class_embed


class CheckpointStructureTests(unittest.TestCase):
    def test_classifies_read_finetuned_single_linear(self):
        keys = {
            "transformer.decoder.class_embed.weight",
            "transformer.decoder.class_embed.bias",
            "class_embed.0.weight",
        }
        self.assertEqual(
            classify_decoder_class_embed(keys, 6),
            {"decoder_class_embed": "single-linear", "decoder_layers": 6},
        )

    def test_classifies_complete_layer_indexed_layout(self):
        keys = {
            f"transformer.decoder.class_embed.{index}.{name}"
            for index in range(6)
            for name in ("weight", "bias")
        }
        self.assertEqual(
            classify_decoder_class_embed(keys, 6),
            {
                "decoder_class_embed": "layer-indexed-module-list",
                "decoder_layers": 6,
            },
        )

    def test_rejects_partial_or_ambiguous_layouts(self):
        with self.assertRaisesRegex(ValueError, "incomplete singular"):
            classify_decoder_class_embed(
                {"transformer.decoder.class_embed.weight"}, 6
            )
        with self.assertRaisesRegex(ValueError, "indices do not match"):
            classify_decoder_class_embed(
                {
                    "transformer.decoder.class_embed.0.weight",
                    "transformer.decoder.class_embed.0.bias",
                },
                6,
            )
        with self.assertRaisesRegex(ValueError, "both singular and indexed"):
            classify_decoder_class_embed(
                {
                    "transformer.decoder.class_embed.weight",
                    "transformer.decoder.class_embed.bias",
                    "transformer.decoder.class_embed.0.weight",
                    "transformer.decoder.class_embed.0.bias",
                },
                6,
            )


if __name__ == "__main__":
    unittest.main()
