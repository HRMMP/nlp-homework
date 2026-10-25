"""Tests for trainable-parameter accounting without loading a real model."""

import unittest
from types import SimpleNamespace

from lora_math_finetuning.model import parameter_stats


class ParameterTests(unittest.TestCase):
    def test_counts_only_requires_grad_parameters(self):
        model = SimpleNamespace(
            named_parameters=lambda: iter([
                ("base.weight", SimpleNamespace(numel=lambda: 100, requires_grad=False)),
                ("lora_A.weight", SimpleNamespace(numel=lambda: 10, requires_grad=True)),
            ])
        )
        stats = parameter_stats(model)
        self.assertEqual(stats["total_parameters"], 110)
        self.assertEqual(stats["trainable_parameters"], 10)
        self.assertAlmostEqual(stats["trainable_ratio"], 10 / 110)
        self.assertEqual(stats["trainable_parameter_names"], ["lora_A.weight"])


if __name__ == "__main__":
    unittest.main()
