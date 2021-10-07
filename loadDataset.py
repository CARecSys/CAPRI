import os
from utils import logger
from config import datasets, dataDirectory


def loadDataset(datasetName):
    datasetFiles = {}
    # If model was not among supported models
    supportedDatasets = list(datasets.keys())
    if not datasetName in supportedDatasets:
        logger('The selected dataset was not among the supported dataset!', 'error')
        return
    # Otherwise, list the files existing in dataset directories
    print(f'Preparing {datasetName} dataset items ...')
    fileList = os.listdir(f'{dataDirectory}/{datasetName}')
    # Create a dictionary of dataset items
    for file in fileList:
        fileName = file.split('.')[0]
        filePath = f'{dataDirectory}\\{datasetName}\\{file}'
        # Add the absolute path to the items: e.g. 'checkins.txt' ==> 'C\\...\\Datacheckins.txt'
        datasetFiles[fileName] = filePath
    return datasetFiles
