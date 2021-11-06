import numpy as np
from utils import logger, textToOperator
from Evaluations.metrics.accuracy import precisionk, recallk
from Models.GeoSoCa.lib.SocialCorrelation import SocialCorrelation
from Models.GeoSoCa.lib.CategoricalCorrelation import CategoricalCorrelation
from Models.GeoSoCa.lib.AdaptiveKernelDensityEstimation import AdaptiveKernelDensityEstimation
from Models.utils import readPoiCoos, readTestData, readCategoryData, readTrainingData, readFriendData, saveModel, loadModel


class GeoSoCaMain:
    def main(datasetFiles, parameters):
        logger('Started processing in GeoSoCa model ...')
        # Reading data from selected dataset
        numberOfUsers, numberOfPoI, numberOfCategories = open(datasetFiles['dataSize'], 'r').readlines()[
            0].strip('\n').split()
        numberOfUsers, numberOfPoI, numberOfCategories = int(
            numberOfUsers), int(numberOfPoI), int(numberOfCategories)
        usersList = list(range(numberOfUsers))
        poiList = list(range(numberOfPoI))
        np.random.shuffle(usersList)
        # Init values
        alpha = 0.5
        modelName = 'GeoSoCa'
        topK = parameters['topK']
        fusion = parameters['fusion']
        datasetName = parameters['datasetName']
        topRestricted = parameters['topRestricted']
        sparsityRatio = parameters['sparsityRatio']
        precision, recall = [], []
        SCScores = np.zeros((numberOfUsers, numberOfPoI))
        CCScores = np.zeros((numberOfUsers, numberOfPoI))
        AKDEScores = np.zeros((numberOfUsers, numberOfPoI))
        # Load libraries
        AKDE = AdaptiveKernelDensityEstimation(alpha)
        SC = SocialCorrelation()
        CC = CategoricalCorrelation()
        logger('Reading dataset instances ...')
        # Loading training items
        trainingMatrix = readTrainingData(
            datasetFiles['train'], numberOfUsers, numberOfPoI, True)
        socialRelations = readFriendData(
            datasetFiles['socialRelations'], 'ndarray', numberOfUsers)
        groundTruth = readTestData(datasetFiles['test'])
        poiCoos = readPoiCoos(datasetFiles['poiCoos'])
        poiCategoryMatrix = readCategoryData(
            datasetFiles['poiCategories'], numberOfCategories, numberOfPoI)
        # Adaptive Kernel Density Estimation Calculations
        AKDE.precomputeKernelParameters(trainingMatrix, poiCoos)
        # Social Correlation Calculations
        loadedModel = loadModel(modelName, datasetName, 'Beta')
        if loadedModel == []:  # It should be created
            SC.computeBeta(trainingMatrix, socialRelations)
            saveModel(SC.X, modelName, datasetName, 'Beta')
        else:  # It should be loaded
            SC.loadModel(loadedModel)
        # Category Correlation Calculations
        loadedModel = loadModel(modelName, datasetName, 'Gamma')
        if loadedModel == []:  # It should be created
            CC.computeGamma(trainingMatrix, poiCategoryMatrix)
            saveModel(CC.Y, modelName, datasetName, 'Gamma')
        else:  # It should be loaded
            CC.loadModel(loadedModel)
        # Add caching policy (prevent a similar setting to be executed again)
        executionRecord = open(
            f"./Generated/GeoSoCa_{datasetName}_top" + str(topRestricted) + ".txt", 'w+')
        # Processing items
        # usersList = usersList[0:9]  # ------------ Temp ------------
        print("Preparing Adaptive Kernel Density Estimation matrix ...")
        loadedModel = loadModel(modelName, datasetName,
                                f'AKDE_{sparsityRatio}')
        if loadedModel == []:  # It should be created
            for cnt, uid in enumerate(usersList):
                if uid in groundTruth:
                    for lid in poiList:
                        AKDEScores[uid, lid] = AKDE.predict(uid, lid)
            saveModel(AKDEScores, modelName, datasetName,
                      f'AKDE_{sparsityRatio}')
        else:  # It should be loaded
            AKDEScores = loadedModel
        logger('Preparing Social Correlation matrix ...')
        loadedModel = loadModel(modelName, datasetName,
                                f'SC_{sparsityRatio}')
        if loadedModel == []:  # It should be created
            for cnt, uid in enumerate(usersList):
                if uid in groundTruth:
                    for lid in poiList:
                        SCScores[uid, lid] = SC.predict(uid, lid)
            saveModel(SCScores, modelName, datasetName,
                      f'SC_{sparsityRatio}')
        else:  # It should be loaded
            SCScores = loadedModel
        logger('Preparing Categorical Correlation matrix ...')
        loadedModel = loadModel(modelName, datasetName,
                                f'CC_{sparsityRatio}')
        if loadedModel == []:  # It should be created
            for cnt, uid in enumerate(usersList):
                if uid in groundTruth:
                    for lid in poiList:
                        CCScores[uid, lid] = CC.predict(uid, lid)
            saveModel(CCScores, modelName, datasetName,
                      f'CC_{sparsityRatio}')
        else:  # It should be loaded
            CCScores = loadedModel
        # Evaluating
        logger('Evaluating results ...')
        for cnt, uid in enumerate(usersList):
            if uid in groundTruth:
                overallScores = [textToOperator(fusion, [AKDEScores[uid, lid], SCScores[uid, lid], CCScores[uid, lid]])
                                 if trainingMatrix[uid, lid] == 0 else -1
                                 for lid in poiList]
                overallScores = np.array(overallScores)
                predicted = list(reversed(overallScores.argsort()))[
                    :topRestricted]
                actual = groundTruth[uid]
                precision.append(precisionk(actual, predicted[:topK]))
                recall.append(recallk(actual, predicted[:topK]))
                print(cnt, uid, f"Precision@{topK}:", '{:.4f}'.format(np.mean(precision)),
                      f", Recall@{topK}:", '{:.4f}'.format(np.mean(recall)))
                executionRecord.write('\t'.join([
                    str(cnt),
                    str(uid),
                    ','.join([str(lid) for lid in predicted])
                ]) + '\n')
