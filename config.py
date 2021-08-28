import os

# Paths
dataDirectory = os.path.abspath('./Data/')
cachingDirectory = os.path.abspath('./Generated/')

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
