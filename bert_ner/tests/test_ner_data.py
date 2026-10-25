import unittest

from ner_data import align_subword_labels, to_bio


class NerDataTests(unittest.TestCase):
    def test_character_offsets_convert_to_bio(self):
        chars, labels, ignored = to_bio(
            "张三在北京大学",
            {"name": {"张三": [[0, 1]]}, "organization": {"北京大学": [[3, 6]]}},
        )
        self.assertEqual("".join(chars), "张三在北京大学")
        self.assertEqual(
            labels,
            [
                "B-name",
                "I-name",
                "O",
                "B-organization",
                "I-organization",
                "I-organization",
                "I-organization",
            ],
        )
        self.assertEqual(ignored, 0)

    def test_longest_nested_span_is_kept_and_counted(self):
        _, labels, ignored = to_bio(
            "《招商银行》",
            {"book": {"《招商银行》": [[0, 5]]}, "company": {"招商银行": [[1, 4]]}},
        )
        self.assertEqual(labels[0], "B-book")
        self.assertEqual(ignored, 1)

    def test_only_first_subword_receives_label(self):
        aligned = align_subword_labels([None, 0, 0, 1, None], [3, 4])
        self.assertEqual(aligned, [-100, 3, -100, 4, -100])


if __name__ == "__main__":
    unittest.main()
