from loadDataset import loadDataset
from argParser import validateUserItems


def __init__():
    print("Welcome! Here you can select the desired sources:")
    userInputs = validateUserItems()
    if (userInputs != None):
        # Initialize dataset items
        datasetFiles = loadDataset(userInputs['Dataset'])
        # Dynamically load Model modules
        module = __import__(
            'Models.' + userInputs['Model'] + '.main', fromlist=[''])
        selectedModule = getattr(module, userInputs['Model'] + 'Main')
        # From the model, select its 'main' class
        selectedModule.main(datasetFiles, userInputs['Dataset'])


__init__()
