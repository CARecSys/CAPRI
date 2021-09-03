from loadDataset import loadDataset
from argParser import validateUserItems
from config import topK, topRestricted, sparsityRatio


def __init__():
    print("Welcome! Here you can select the desired sources:")
    userInputs = validateUserItems()
    if (userInputs != None):
        # Initialize dataset items
        datasetFiles = loadDataset(userInputs['Dataset'])
        # Initializing parameters
        parameters = {
            "topK": topK,
            "topRestricted": topRestricted,
            "sparsityRatio": sparsityRatio,
            "datasetName": userInputs['Dataset'],
        }
        # Dynamically load Model modules
        module = __import__(
            'Models.' + userInputs['Model'] + '.main', fromlist=[''])
        selectedModule = getattr(module, userInputs['Model'] + 'Main')
        # From the model, select its 'main' class
        selectedModule.main(datasetFiles, parameters)


__init__()
