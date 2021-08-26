import os
from config import datasets, dataDirectory


def loadDataset(datasetName):
    # If model was not among supported models
    supportedDatasets = list(datasets.keys())
    if not datasetName in supportedDatasets:
        print('[Error] Dataset not supported!')
        return
    # Otherwise, list the files existing in dataset directories
    print(f'Preparing {datasetName} dataset items ...')
    datasetFiles = os.listdir(f'{dataDirectory}/{datasetName}')
    print(datasetFiles)


loadDataset('Yelp')
