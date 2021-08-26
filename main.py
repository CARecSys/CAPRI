from loadDataset import loadDataset
from argParser import validateUserItems


def __init__():
    print("Welcome! Here you can select the desired sources:")
    userInputs = validateUserItems()
    if (userInputs != None):
        # Initialize dataset items
        loadDataset(userInputs['Dataset'])


__init__()
