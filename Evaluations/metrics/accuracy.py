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
    Computes the Normalized Discounted Cumulative Gain (NDCG) for top k items in the ranked output

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
    idcg = 1.0
    dcg = 1.0 if predicted[0] in actual else 0.0
    for i, p in enumerate(predicted[1:]):
        if p in actual:
            dcg += 1.0 / np.log(i+2)
        idcg += 1.0 / np.log(i+2)
    ndcg = dcg / idcg
    return ndcg


def ndcg(rankedList: list, itemsPositions, relevance=None):
    """
    Computes the Normalized Discounted Cumulative Gain (NDCG) for top k items in the ranked output

    Parameters
    ----------
    rankedList: list
        -
        example: [X, Y, Z]
    itemsPositions: list
        -
        example: [x, y, z]
    relevance:
        -

    Returns
    ----------
    ndcg:
        Normalized DCG score

    Metric Defintion
    ----------
    Jarvelin, K., & Kekalainen, J. (2002). Cumulated gain-based evaluation of IR techniques.
    ACM Transactions on Information Systems (TOIS), 20(4), 422-446.
    """
    if relevance is None:
        relevance = np.ones_like(itemsPositions)
    assert len(relevance) == itemsPositions.shape[0]
    # A dictionary for mapping the id of items to their corresponding relevance
    mappedDictionary = {it: r for it, r in zip(itemsPositions, relevance)}
    rankScores = np.asarray([mappedDictionary.get(it, 0.0)
                            for it in rankedList[:None]], dtype=np.float32)
    # Calculating ideal DCG
    scores = np.sort(relevance)[::-1]
    idealDCG = np.sum(np.divide(np.power(2, scores) - 1, np.log(np.arange(scores.shape[0], dtype=np.float32) + 2)),
                      dtype=np.float32)
    rankDCG = np.sum(np.divide(np.power(2, rankScores) - 1, np.log(np.arange(rankScores.shape[0], dtype=np.float32) + 2)),
                     dtype=np.float32)
    if rankDCG == 0.0:
        return 0.0
    ndcg = rankDCG / idealDCG
    return ndcg
