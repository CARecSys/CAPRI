# Precision
# Description: Correct Positive Predictions over Total Positive Predictions (TP / TP+FP)

def precisionk(actual, predicted):
    return 1.0 * len(set(actual) & set(predicted)) / len(predicted)
