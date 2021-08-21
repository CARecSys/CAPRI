import sys
from argParser import validateUserItems


def __init__():
    print("Welcome! Here you can select the desired sources:")
    userInputs = validateUserItems()
    if (userInputs != None):
        print('Rest of the code')


__init__()
