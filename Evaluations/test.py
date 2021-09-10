import unittest
import numpy as np
from metrics import precisionk, recallk, listDiversity, mapk, ndcgk, novelty, catalogCoverage, personalization


class TestMetrics(unittest.TestCase):
    # MapK
    def test_mapk_joint(self):
        expected = 1.0
        actual = mapk(['1', '5', '7', '9'], ['1', '5', '7', '9'], 10)
        self.assertEqual(actual, expected)

    def test_mapk_coverage(self):
        expected = 0.75
        actual = mapk(range(1, 5), range(1, 4), 10)
        self.assertEqual(actual, expected)

    def test_mapk_disjoint(self):
        expected = 0.0
        actual = mapk([1, 2, 3, 4], [5, 6, 7], 10)
        self.assertEqual(actual, expected)

    # NDCG (TODO: check again)
    def test_ndcgk_correct(self):
        expected = 0.980840401274087
        actual = ndcgk(np.asarray([3, 2, 1, 0, 0]),
                       np.asarray([3, 2, 0, 0, 1]), 1)
        self.assertEqual(actual, expected)

    # Precision (TODO: check again)
    def test_precision_correct(self):
        expected = 0.5
        actual = precisionk([1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1])
        self.assertEqual(actual, expected)

    # Recall
    def test_recall_correct(self):
        expected = 0.6666667
        actual = recallk([1, 1, 1, 0, 0, 0], [1, 0, 1, 1, 1, 1])
        self.assertEqual(actual, expected)

    # Diversity
    def test_diversity_correct(self):
        expected = 0.0
        actual = listDiversity([1, 2, 3], [1, 2])
        self.assertEqual(actual, expected)

    # Novelty
    def test_novelty_correct(self):
        predicted = ['X', 'Y', 'Z']
        pop = {1198: 893, 1270: 876, 593: 876, 2762: 867}
        numberOfUsers = 100
        listLength = 10
        process = novelty(predicted, pop, numberOfUsers, listLength)
        expected = 10
        self.assertEqual(process, expected)

    # Catalog Coverage
    def test_catalogCoverage_correct(self):
        expected = 83.33
        actual = catalogCoverage(
            [['A', 'B', 'C', 'D'], ['A', 'B', 'C', 'Z']], ['A', 'B', 'C', 'X', 'Y', 'Z'])
        self.assertEqual(actual, expected)

    # Personalization
    def test_personalization_correct(self):
        expected = 0.25
        actual = personalization(
            [['A', 'B', 'C', 'D'], ['A', 'B', 'C', 'X'], ['A', 'B', 'C', 'Z']])
        self.assertEqual(actual, expected)
