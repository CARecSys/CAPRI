import unittest
import numpy as np
from metrics.accuracy import precisionk, recallk, mapk, ndcgk
from metrics.beyoundAccuracy import listDiversity, novelty, catalogCoverage, personalization


class TestMetrics(unittest.TestCase):
    # -------------------- Accuracy Metrics ------------------
    # Map-K
    def test_mapk_joint(self):
        k = 10
        actual = ['1', '5', '7', '9']
        predicted = ['1', '5', '7', '9']
        expected = 1.0
        calculated = mapk(actual, predicted, k)
        self.assertEqual(calculated, expected)

    def test_mapk_coverage(self):
        k = 5
        actual = range(1, 5)
        predicted = range(1, 4)
        expected = 0.75
        calculated = mapk(actual, predicted, k)
        self.assertEqual(calculated, expected)

    def test_mapk_disjoint(self):
        k = 2
        actual = [1, 2, 3, 4]
        predicted = [5, 6, 7]
        expected = 0.0
        calculated = mapk(actual, predicted, k)
        self.assertEqual(calculated, expected)

    # Precision
    def test_precision_correct(self):
        actual = [1, 2, 3, 4, 5, 6]
        predicted = [1, 2, 3, 7, 8, 9]
        expected = 0.5
        calculated = precisionk(actual, predicted)
        self.assertEqual(calculated, expected)

    # Recall
    def test_recall_correct(self):
        actual = [1, 2, 3, 4, 5, 0]
        predicted = [1, 2, 3, 7, 8, 9]
        expected = 0.5
        calculated = recallk(actual, predicted)
        self.assertEqual(calculated, expected)

    # NDCG
    def test_ndcgk_correct(self):
        actual = [3, 2, 1, 0, 0]
        predicted = [3, 2, 0, 0, 1]
        expected = 1.0
        calculated = ndcgk(actual, predicted)
        self.assertEqual(calculated, expected)

    # -------------------- Beyound-Accuracy Metrics ------------------
    # Diversity
    def test_diversity_correct(self):
        predicted = [0, 0, 0]
        itemsSimilarityMatrix = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 3, 1]])
        expected = 0.0
        calculated = listDiversity(predicted, itemsSimilarityMatrix)
        self.assertEqual(calculated, expected)

    # Novelty
    def test_novelty_correct(self):
        predicted = [1, 2, 3]
        pop = {1: 10, 2: 20, 3: 30}
        numberOfUsers = 100
        listLength = 10
        expected = 0.74
        calculated = novelty(predicted, pop, numberOfUsers, listLength)
        self.assertAlmostEqual(calculated, expected, 2)

    # Catalog Coverage
    def test_catalogCoverage_correct(self):
        predicted = [['A', 'B', 'C', 'D'], ['A', 'B', 'C', 'Z']]
        catalog = ['A', 'B', 'C', 'X', 'Y', 'Z']
        expected = 83.3
        calculated = catalogCoverage(predicted, catalog)
        self.assertAlmostEqual(calculated, expected, 1)

    # Personalization
    def test_personalization_correct(self):
        predicted = [['A', 'B', 'C', 'D'], [
            'A', 'B', 'C', 'X'], ['A', 'B', 'C', 'Z']]
        expected = 0.25
        calculated = personalization(predicted)
        self.assertEqual(calculated, expected)
