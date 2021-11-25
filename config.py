import os

# Paths
dataDirectory = os.path.abspath('./Data/')
evaluationResultsDir = os.path.abspath('./Generated/')

# Default Parameters
topK = 10
topRestricted = 100  # Restricted list for computation
sparsityRatio = 100

# Key: Model name, Value: Covered Contexts
models = {
    "GeoSoCa": ["Geographical", "Social", "Categorical"],
    "LORE": ["Geographical", "Social", "Temporal"],
    "USG": ["Interaction", "Social", "Geographical"],
}

# Key: Dataset name, Value: Covered Contexts
datasets = {
    "Gowalla": ["Geographical", "Social", "Temporal"],
    "Yelp":  ["Geographical", "Social", "Temporal", "Categorical"],
}

# An array of selected operations
fusions = ["Product", "Sum", "WeightedSum"]

# List of evaluation metrics
evaluationMetrics = ["Precision", "Recall", "mAP", "NDCG"]

# Models Dictionaries
GeoSoCaDict = {"alpha": 0.5}
USGDict = {"alpha": 0.1, "beta": 0.1}
LoreDict = {"alpha": 0.05, "deltaT": 3600 * 24}
