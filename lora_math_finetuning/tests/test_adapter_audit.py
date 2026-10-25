"""Saved tensor headers must match rank, target modules and recorded parameters."""

import unittest

from lora_math_finetuning.audit_adapter import check_tensors


class AdapterAuditTests(unittest.TestCase):
    def setUp(self):
        self.model = {
            "num_hidden_layers": 1,
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
        }
        self.config = {"r": 2, "target_modules": ["q_proj", "v_proj"]}
        self.shapes = {
            f"base_model.model.model.layers.0.self_attn.{module}.lora_{factor}.weight": shape
            for module, factor, shape in (
                ("q_proj", "A", [2, 8]),
                ("q_proj", "B", [8, 2]),
                ("v_proj", "A", [2, 8]),
                ("v_proj", "B", [4, 2]),
            )
        }
        self.stats = {
            "trainable_parameter_names": list(self.shapes),
            "trainable_parameters": 56,
            "total_parameters": 1000,
        }

    def test_counts_and_target_coverage(self):
        result = check_tensors(self.shapes, self.stats, self.config, self.model)
        self.assertEqual(result["trainable_parameters"], 56)
        self.assertEqual(result["tensor_count"], 4)

    def test_rejects_wrong_rank_shape(self):
        shapes = dict(self.shapes)
        shapes[next(iter(shapes))] = [3, 8]
        with self.assertRaises(ValueError):
            check_tensors(shapes, self.stats, self.config, self.model)

    def test_rejects_count_or_name_mismatch(self):
        with self.assertRaises(ValueError):
            check_tensors(
                self.shapes,
                {**self.stats, "trainable_parameters": 100},
                self.config,
                self.model,
            )
        with self.assertRaises(ValueError):
            check_tensors(
                {**self.shapes, "unexpected": [2, 8]},
                self.stats,
                self.config,
                self.model,
            )


if __name__ == "__main__":
    unittest.main()
