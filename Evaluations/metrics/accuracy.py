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
        A list of items recommended by the system
        example: [x, y, z]
    ----------
    Returns
        precision at k
    """
    relevantResults = set(actual) & set(recommended)
    assert 0 <= len(
        relevantResults), f"The number of relevant results is not true (currently: {len(relevantResults)})"
    return 1.0 * len(relevantResults) / len(recommended)


def recallk(actual: list, recommended: list):
    """
    Computes the correct Positive Predictions over Actual Positive Values (TP / TP+FN)
    Parameters
    ----------
    actual: list
        A list of ground truth items
        example: [X, Y, Z]
    recommended: list
        A list of items recommended by the system
        example: [x, y, z]
    ----------
    Returns
        recall at k
    """
    relevantResults = set(actual) & set(recommended)
    assert 0 <= len(
        relevantResults), f"The number of relevant results is not true (currently: {len(relevantResults)})"
    return 1.0 * len(relevantResults) / len(actual)


def mapk(actual: list, predicted: list, k: int):
    """
    Computes mean Average Precision at k (mAPk) for a set of predictions
    Parameters
    ----------
    actual: list
        A list of ground truth items
        example: [X, Y, Z]
    predicted: list
        A list of predicted items, recommended by the system
        example: [x, y, z]
    k: integer
        The number of elements of predicted to consider in the calculation
    ----------
    Returns
    result:
        mean Average Precision at k (mAPk)
    """
    score = 0.0
    numberOfHits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            numberOfHits += 1.0
            score += numberOfHits / (i+1.0)
    if not actual:
        return 0.0
    result = score / min(len(actual), k)
    return result


def ndcgk(actual: list, predicted: list):
    """
    Computes the Normalized Discounted Cumulative Gain (NDCG) a ranking quality analyzer
    Parameters
    ----------
        actual: list
        A list of ground truth items
        example: [X, Y, Z]
    predicted: list
        A list of predicted items, recommended by the system
        example: [x, y, z]
    ----------
    Returns
    ndcg:
        Normalized DCG score
    ----------
    Metric Defintion:
    Jarvelin, K., & Kekalainen, J. (2002). Cumulated gain-based evaluation of IR techniques.
    ACM Transactions on Information Systems (TOIS), 20(4), 422-446.
    """
    idcg = 1.0
    dcg = 1.0 if predicted[0] in actual else 0.0
    for i, p in enumerate(predicted[1:]):
        if p in actual:
            dcg += 1.0 / np.log(i+2)
        idcg += 1.0 / np.log(i+2)
    ndcg = dcg / idcg
    return ndcg
