import logging
import datetime
from loadDataset import loadDataset
from argParser import validateUserItems
from config import topK, topRestricted, sparsityRatio


def __init__():
    print("Welcome! Here you can select the desired sources:")
    # Creating log file
    logging.basicConfig(filename='logger.log', level=logging.INFO)
    currentMoment = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f'[{currentMoment}] Framework started!')
    # Getting inputs from users
    userInputs = validateUserItems()
    # If selections were not empty
    if (userInputs != None):
        # Initialize dataset items
        datasetFiles = loadDataset(userInputs['Dataset'])
        logging.info(f'Dataset files: {datasetFiles}')
        # Initializing parameters
        parameters = {
            "topK": topK,
            "topRestricted": topRestricted,
            "sparsityRatio": sparsityRatio,
            "fusion": userInputs['Fusion'],
            "datasetName": userInputs['Dataset'],
        }
        logging.info(f'Processing parameters: {parameters}')
        # Dynamically load Model modules
        module = __import__(
            'Models.' + userInputs['Model'] + '.main', fromlist=[''])
        selectedModule = getattr(module, userInputs['Model'] + 'Main')
        # From the model, select its 'main' class
        selectedModule.main(datasetFiles, parameters)
    else:
        currentMoment = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logging.info(f'[{currentMoment}] Framework was stopepd!')


__init__()
