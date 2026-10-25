"""Strict entities, mutually exclusive error partitions, and unseen surfaces."""

import unittest

from analyze import classify_error, entity_summary, strict_spans


class EntityAnalysisTests(unittest.TestCase):
    def test_invalid_i_start_is_not_repaired(self):
        self.assertEqual(strict_spans(["I-name", "I-name", "O"]), set())
        self.assertEqual(strict_spans(["B-name", "I-name", "O"]), {(0, 2, "name")})

    def test_positions_map_to_original_characters(self):
        self.assertEqual(strict_spans(["B-name", "I-name"], [2, 4]), {(2, 5, "name")})
        with self.assertRaises(ValueError):
            strict_spans(["O"], [])

    def test_half_open_neighbors_do_not_overlap(self):
        self.assertEqual(
            classify_error((0, 2, "name"), {(2, 4, "name")}), "no_overlapping_entity"
        )

    def test_boundary_and_type_priority_is_explicit(self):
        gold = (0, 2, "name")
        self.assertEqual(
            classify_error(gold, {(0, 2, "company")}), "same_span_wrong_type"
        )
        self.assertEqual(
            classify_error(gold, {(1, 3, "name")}), "same_type_wrong_boundary"
        )
        self.assertEqual(
            classify_error(gold, {(1, 3, "company")}), "overlap_wrong_type_and_boundary"
        )

    def test_partition_counts_and_novelty_are_consistent(self):
        statistics, _ = entity_summary(
            ["abcdef"],
            [{(0, 2, "name"), (3, 5, "company")}],
            [{(0, 2, "name"), (3, 5, "address"), (5, 6, "name")}],
            {("ab", "name")},
        )
        self.assertEqual(statistics["true_positive"], 1)
        self.assertEqual(statistics["false_negative"], 1)
        self.assertEqual(statistics["false_positive"], 2)
        self.assertEqual(
            statistics["same_span_type_confusions"], {"company->address": 1}
        )
        self.assertEqual(
            statistics["surface_type_seen_in_training"]["seen"]["recall"], 1.0
        )
        self.assertEqual(
            statistics["surface_type_seen_in_training"]["unseen"]["gold"], 1
        )
        self.assertEqual(
            statistics["surface_type_seen_in_training"]["unseen"]["correct"], 0
        )

    def test_unmatched_entity_has_only_one_category(self):
        statistics, _ = entity_summary(
            ["abcdef"], [{(0, 3, "name")}], [{(0, 3, "company"), (1, 2, "name")}], set()
        )
        self.assertEqual(
            statistics["false_negative_categories"], {"same_span_wrong_type": 1}
        )
        self.assertEqual(sum(statistics["false_positive_categories"].values()), 2)


if __name__ == "__main__":
    unittest.main()
