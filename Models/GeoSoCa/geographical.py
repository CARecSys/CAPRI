import numpy as np
from config import GeoSoCaDict, sparsityRatio
from Models.utils import loadModel, saveModel
from Models.GeoSoCa.lib.AdaptiveKernelDensityEstimation import AdaptiveKernelDensityEstimation

modelName = 'GeoSoCa'


def geographicalCalculations(datasetName: str, users: dict, pois: dict, poiCoos: dict, trainingMatrix, groundTruth):
    """
    This function calculates the geographical parameters of the dataset

    Parameters
    ----------
    datasetName : str
        The name of the dataset
    users : dict
        The users of the dataset
    pois : dict
        The pois of the dataset
    poiCoos : dict 
        The poi coordinates of the dataset
    groundTruth : dict
        The ground truth of the dataset
    trainingMatrix : dict
        The training matrix of the dataset

    Returns
    -------
    AKDEScores : dict
        The AKDE scores of the dataset
    """
    # Initializing parameters
    alpha = GeoSoCaDict['alpha']
    AKDEScores = np.zeros((users['count'], pois['count']))
    print("Preparing Adaptive Kernel Density Estimation matrix ...")
    loadedModel = loadModel(modelName, datasetName,
                            f'AKDE_{sparsityRatio}')
    if loadedModel == []:  # It should be created
        # Creating object to AKDE Class
        AKDE = AdaptiveKernelDensityEstimation(alpha)
        # Calculating AKDE scores
        # TODO: We may be able to load the model from disk
        AKDE.precomputeKernelParameters(trainingMatrix, poiCoos)
        print("Now, training the model for each user ...")
        for counter, uid in enumerate(users['list']):
            # Adding log to console
            if (counter % 100 == 0):
                print(f'{counter} users processed ...')
            if uid in groundTruth:
                for lid in pois['list']:
                    AKDEScores[uid, lid] = AKDE.predict(uid, lid)
        saveModel(AKDEScores, modelName, datasetName,
                  f'AKDE_{sparsityRatio}')
    else:  # It should be loaded
        AKDEScores = loadedModel
    # Returning the scores
    return AKDEScores
