"""CPU checks for image-token insertion and answer-only supervision."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from src.data import PLACEHOLDER_ID, conversations, encode_messages
from src.model import ConstructedLVLM, build_adapter


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


class FakeLanguage(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(256, 3)

    def get_input_embeddings(self):
        return self.embedding


class PipelineContracts(unittest.TestCase):
    def test_adapter_options_have_same_output_shape_and_distinct_parameters(self):
        linear = build_adapter(1024, 896, "linear")
        mlp = build_adapter(1024, 896, "mlp")
        features = torch.randn(2, 3, 1024)
        for adapter in (linear, mlp):
            output = adapter(features)
            self.assertEqual(tuple(output.shape), (2, 3, 896))
            output.square().mean().backward()
            self.assertTrue(all(parameter.grad is not None for parameter in adapter.parameters()))
        self.assertGreater(sum(p.numel() for p in mlp.parameters()),
                           sum(p.numel() for p in linear.parameters()))

    def test_checkpoint_rejects_different_adapter_type(self):
        with TemporaryDirectory() as directory:
            linear = ConstructedLVLM.__new__(ConstructedLVLM)
            nn.Module.__init__(linear)
            linear.config_values = {"adapter_type": "linear"}
            linear.adapter = build_adapter(1024, 896, "linear")
            checkpoint = Path(directory) / "checkpoint"
            linear.save_state(checkpoint, include_language=False,
                              metadata={"config": linear.config_values})

            mlp = ConstructedLVLM.__new__(ConstructedLVLM)
            nn.Module.__init__(mlp)
            mlp.config_values = {"adapter_type": "mlp"}
            mlp.adapter = build_adapter(1024, 896, "mlp")
            with self.assertRaisesRegex(ValueError, "adapter type"):
                mlp.load_state(checkpoint, include_language=False)

    def test_assistant_only_labels_and_single_image_marker(self):
        record = {"conversations": [
            {"from": "human", "value": "Describe this.\n<image>"},
            {"from": "gpt", "value": "A dog."},
        ]}
        ids, labels = encode_messages(CharacterTokenizer(), conversations(record))
        self.assertEqual(ids.count(PLACEHOLDER_ID), 1)
        position = ids.index(PLACEHOLDER_ID)
        self.assertEqual(labels[position], -100)
        self.assertTrue(all(label == -100 for label in labels[:position]))
        self.assertIn(ord("d"), [label for label in labels if label != -100])

    def test_rejects_multiple_images(self):
        record = {"conversations": [
            {"from": "human", "value": "<image> and <image>"},
            {"from": "gpt", "value": "Two images."},
        ]}
        with self.assertRaises(ValueError):
            conversations(record)

    def test_visual_patches_replace_only_marker_and_keep_labels_masked(self):
        model = ConstructedLVLM.__new__(ConstructedLVLM)
        nn.Module.__init__(model)
        model.language = FakeLanguage()
        patches = torch.ones(2, 3)
        embeddings, attention, labels = model.assemble(
            [[10, PLACEHOLDER_ID, 11], [12, PLACEHOLDER_ID, 13, 14]],
            [[-100, -100, 11], [-100, -100, 13, 14]],
            torch.stack((patches, patches)),
        )
        self.assertEqual(tuple(embeddings.shape), (2, 5, 3))
        self.assertEqual(attention.tolist(), [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
        self.assertEqual(labels.tolist(), [[-100, -100, -100, 11, -100],
                                           [-100, -100, -100, 13, 14]])


if __name__ == "__main__":
    unittest.main()
