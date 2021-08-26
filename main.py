from loadDataset import loadDataset
from argParser import validateUserItems


def __init__():
    print("Welcome! Here you can select the desired sources:")
    userInputs = validateUserItems()
    if (userInputs != None):
        # Initialize dataset items
        datasetFiles = loadDataset(userInputs['Dataset'])
        print(datasetFiles)


__init__()
