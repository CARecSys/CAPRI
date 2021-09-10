import unittest
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

    # # Precision
    # def test_precision_correct(self):
    #     expected = [2]
    #     actual = precisionk([1], [2])
    #     self.assertEqual(actual, expected)

    # # Recall
    # def test_recall_correct(self):
    #     expected = [2]
    #     actual = recallk([1], [2])
    #     self.assertEqual(actual, expected)

    # # NDCG
    # def test_ndcgk_correct(self):
    #     expected = [2]
    #     actual = ndcgk([1], [2], 10)
    #     self.assertEqual(actual, expected)

    # # Diversity
    # def test_diversity_correct(self):
    #     expected = [2]
    #     actual = listDiversity([1], [2], 10)
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
    #     expected = [2]
    #     actual = catalogCoverage([1], [2], 10)
    #     self.assertEqual(actual, expected)

    # # Personalization
    # def test_personalization_correct(self):
    #     expected = [2]
    #     actual = personalization([1], [2], 10)
    #     self.assertEqual(actual, expected)
