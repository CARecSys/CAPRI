# Recall
# Description: Correct Positive Predictions over Actual Positive Values (TP / TP+FN)

def recallk(actual, predicted):
    return 1.0 * len(set(actual) & set(predicted)) / len(actual)
