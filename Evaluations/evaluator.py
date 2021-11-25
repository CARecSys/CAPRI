import numpy as np
from Models.utils import normalize
from utils import logger, textToOperator
from Evaluations.metrics.accuracy import precisionk, recallk
from config import USGDict, topK, topRestricted, evaluationResultsDir


def overallScoreCalculator(modelName: str, userId, evalParams, modelParams):
    """
    Calculate the overall score of the model based on the given parameters

    Parameters
    ----------
    modelName : str
        Name of the model to be evaluated
    userId : int
        User ID
    evalParams : dict
        Dictionary of evaluation parameters
    modelParams : dict
        Dictionary of model parameters

    Returns
    -------
    overallScores : numpy.ndarray
        Array of overall scores
    """
    # Extracting the list of parameters
    fusion, poiList, trainingMatrix = evalParams['fusion'], evalParams['poiList'], evalParams['trainingMatrix']
    # Checking for proper model
    if (modelName == 'GeoSoCa'):
        AKDEScores, SCScores, CCScores = modelParams['AKDE'], modelParams['SC'], modelParams['CC']
        overallScores = [textToOperator(fusion, [AKDEScores[userId, lid], SCScores[userId, lid], CCScores[userId, lid]])
                         if trainingMatrix[userId, lid] == 0 else -1
                         for lid in poiList]
    elif (modelName == 'LORE'):
        KDEScores, FCFScores, AMCScores = modelParams['KDE'], modelParams['FCF'], modelParams['AMC']
        overallScores = [textToOperator(fusion, [KDEScores[userId, lid], FCFScores[userId, lid], AMCScores[userId, lid]])
                         if (userId, lid) not in trainingMatrix else -1
                         for lid in poiList]
    elif (modelName == 'USG'):
        alpha, beta = USGDict['alpha'], USGDict['beta']
        UScores, SScores, GScores = modelParams['U'], modelParams['S'], modelParams['G']
        U_scores = normalize([UScores[userId, lid]
                              if trainingMatrix[userId, lid] == 0 else -1
                              for lid in poiList])
        S_scores = normalize([SScores[userId, lid]
                              if trainingMatrix[userId, lid] == 0 else -1
                              for lid in poiList])
        G_scores = normalize([GScores[userId, lid]
                              if trainingMatrix[userId, lid] == 0 else -1
                              for lid in poiList])
        U_scores, S_scores, G_scores = np.array(
            U_scores), np.array(S_scores), np.array(G_scores)
        overallScores = textToOperator(
            fusion, [(1.0 - alpha - beta) * U_scores, alpha * S_scores, beta * G_scores])
    return np.array(overallScores)


def evaluator(modelName: str, datasetName: str, evalParams: dict, modelParams: dict):
    """
    Evaluate the model with the given parameters and return the evaluation metrics

    Parameters
    ----------
    modelName : str
        Name of the model to be evaluated
    datasetName : str
        Name of the dataset to be evaluated
    evalParams : dict
        Dictionary of evaluation parameters
    modelParams : dict
        Dictionary of model parameters
    """
    logger('Evaluating results ...')
    # Fetching the list of parameters
    usersList, groundTruth, fusion = evalParams['usersList'], evalParams['groundTruth'], evalParams['fusion']
    # Initializing the metrics
    precision, recall, map, ndcg = [], [], [], []
    # Add caching policy (prevent a similar setting to be executed again)
    fileName = f'Eval_{modelName}_{datasetName}_{fusion}_top{topK}_limit{topRestricted}'
    evaluationResults = open(f"./{evaluationResultsDir}{fileName}.txt", 'w+')
    for counter, userId in enumerate(usersList):
        if userId in groundTruth:
            overallScores = []
            # Processing items
            overallScores = overallScoreCalculator(
                modelName, userId, evalParams, modelParams)
            predicted = list(reversed(overallScores.argsort()))[
                :topRestricted]
            actual = groundTruth[userId]
            precision.append(precisionk(actual, predicted[:topK]))
            recall.append(recallk(actual, predicted[:topK]))
            print(counter, userId, f"Precision@{topK}:", '{:.4f}'.format(np.mean(precision)),
                  f", Recall@{topK}:", '{:.4f}'.format(np.mean(recall)))
            evaluationResults.write('\t'.join([
                str(counter),
                str(userId),
                ','.join([str(lid) for lid in predicted])
            ]) + '\n')
    evaluationResults.close()
