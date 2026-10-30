from __future__ import annotations

import random
import unittest

from evaluation.evaluate import corrupt_sequence


class EvaluationTests(unittest.TestCase):
    def test_insertions_are_not_only_appended_and_cropped(self) -> None:
        original = list(range(1, 101))
        modified = corrupt_sequence(original, 0.10, "insert", random.Random(7))
        self.assertEqual(len(modified), len(original))
        first_difference = next(i for i, (x, y) in enumerate(zip(original, modified)) if x != y)
        self.assertLess(first_difference, 50)
        self.assertTrue(all(0 <= value <= 255 for value in modified))


if __name__ == "__main__":
    unittest.main()
