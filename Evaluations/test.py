import unittest
import numpy as np
from metricsAccuracy import precisionk, recallk, mapk, ndcgk
from metricsBeyoundAccuracy import listDiversity, novelty, catalogCoverage, personalization


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
    # # Diversity
    # def test_diversity_correct(self):
    #     expected = 0.0
    #     actual = listDiversity([1, 2, 3], [1, 2])
    #     self.assertEqual(actual, expected)

    # # Novelty
    # def test_novelty_correct(self):
    #     predicted = ['X', 'Y', 'Z']
    #     pop = {1198: 893, 1270: 876, 593: 876, 2762: 867}
    #     numberOfUsers = 100
    #     listLength = 10
    #     process = novelty(predicted, pop, numberOfUsers, listLength)
    #     expected = 10
    #     self.assertEqual(process, expected)

    # # Catalog Coverage
    # def test_catalogCoverage_correct(self):
    #     expected = 83.33
    #     actual = catalogCoverage(
    #         [['A', 'B', 'C', 'D'], ['A', 'B', 'C', 'Z']], ['A', 'B', 'C', 'X', 'Y', 'Z'])
    #     self.assertEqual(actual, expected)

    # # Personalization
    # def test_personalization_correct(self):
    #     expected = 0.25
    #     actual = personalization(
    #         [['A', 'B', 'C', 'D'], ['A', 'B', 'C', 'X'], ['A', 'B', 'C', 'Z']])
    #     self.assertEqual(actual, expected)
