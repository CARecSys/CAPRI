import logging
from utils import logger
from loadDataset import loadDataset
from commandParser import getUserChoices


def __init__():
    # Creating log file
    logging.basicConfig(filename='logger.log', level=logging.INFO)
    logger('CAPRI framework started!')
    # Getting inputs from users
    userInputs = getUserChoices()
    # If selections were not empty
    if (userInputs != None):
        # Initialize dataset items
        datasetFiles = loadDataset(userInputs['Dataset'])
        logger(f'Dataset files: {datasetFiles}', 'info', True)
        # Initializing parameters
        parameters = {
            "fusion": userInputs['Fusion'],
            "datasetName": userInputs['Dataset'],
            "evaluation": userInputs['Evaluation'],
        }
        logger(f'Processing parameters: {parameters}', 'info', True)
        # Dynamically load Model modules
        module = __import__(
            'Models.' + userInputs['Model'] + '.main', fromlist=[''])
        selectedModule = getattr(module, userInputs['Model'] + 'Main')
        # From the model, select its 'main' class
        selectedModule.main(datasetFiles, parameters)
    else:
        logger('Framework stopepd!')


__init__()
