import numpy as np


def precisionk(actual: list, recommended: list):
    """
    Computes the number of relevant results among the top k recommended items

    Parameters
    ----------
    actual: list
        A list of ground truth items
        example: [X, Y, Z]
    recommended: list
        A list of ground truth items (all possible relevant items)
        example: [x, y, z]

    Returns
    ----------
        precision at k
    """
    relevantResults = set(actual) & set(recommended)
    assert 0 <= len(
        relevantResults), f"The number of relevant results is not true (currently: {len(relevantResults)})"
    return 1.0 * len(relevantResults) / len(recommended)


def recallk(actual: list, recommended: list):
    """
    The number of relevant results among the top k recommended items divided by the total number of relevant items

    Parameters
    ----------
    actual: list
        A list of ground truth items (all possible relevant items)
        example: [X, Y, Z]
    recommended: list
        A list of items recommended by the system
        example: [x, y, z]

    Returns
    ----------
        recall at k
    """
    relevantResults = set(actual) & set(recommended)
    assert 0 <= len(
        relevantResults), f"The number of relevant results is not true (currently: {len(relevantResults)})"
    return 1.0 * len(relevantResults) / len(actual)


def mapk(actual: list, predicted: list, k: int = 10):
    """
    Computes mean Average Precision at k (mAPk) for a set of recommended items

    Parameters
    ----------
    actual: list
        A list of ground truth items (order doesn't matter)
        example: [X, Y, Z]
    predicted: list
        A list of predicted items, recommended by the system (order matters)
        example: [x, y, z]
    k: integer, optional (default to 10)
        The number of elements of predicted to consider in the calculation

    Returns
    ----------
    score:
        The mean Average Precision at k (mAPk)
    """
    score = 0.0
    numberOfHits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            numberOfHits += 1.0
            score += numberOfHits / (i+1.0)
    if not actual:
        return 0.0
    score = score / min(len(actual), k)
    return score


def ndcgk(actual: list, predicted: list):
    """
    Calculates the implicit version of Normalized Discounted Cumulative Gain (NDCG) for top k items in the ranked output

    Parameters
    ----------
    actual: list
        A list of ground truth items
        example: [X, Y, Z]
    predicted: list
        A list of predicted items, recommended by the system
        example: [x, y, z]

    Returns
    ----------
    ndcg:
        Normalized DCG score

    Metric Defintion
    ----------
    Jarvelin, K., & Kekalainen, J. (2002). Cumulated gain-based evaluation of IR techniques.
    ACM Transactions on Information Systems (TOIS), 20(4), 422-446.
    """
    idcg = 1.0  # the ideal DCG is 1
    # The discounted cumulative gain sets to 1, if the first item of the predicted list exists in the ground-truth
    dcg = 1.0 if predicted[0] in actual else 0.0
    for i, p in enumerate(predicted[1:]):
        # i is the index (0, 1, 2, ...) and p is the remaining predicted elements
        if p in actual:
            # DCG is added by the value of relevance score divided by log, only if it exists in the ground-truth array
            dcg += 1.0 / np.log(i+2)
        # Ideal DCG is added by the value of relevance score divided by log for all predicted elements
        idcg += 1.0 / np.log(i+2)
    # Normalization
    ndcg = dcg / idcg
    return ndcg
