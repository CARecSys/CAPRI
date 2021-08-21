import numpy as np

# Normalized Discounted Cumulative Gain (NDCG)
# Discription: a ranking quality analyzer


def ndcgk(actual, predicted, k):
    idcg = 1.0
    dcg = 1.0 if predicted[0] in actual else 0.0
    for i, p in enumerate(predicted[1:]):
        if p in actual:
            dcg += 1.0 / np.log(i+2)
        idcg += 1.0 / np.log(i+2)
    return dcg / idcg
