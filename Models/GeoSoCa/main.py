import numpy as np
from Evaluations.metrics import precisionk, recallk
from Models.GeoSoCa.lib.SocialCorrelation import SocialCorrelation
from Models.GeoSoCa.lib.CategoricalCorrelation import CategoricalCorrelation
from Models.GeoSoCa.lib.AdaptiveKernelDensityEstimation import AdaptiveKernelDensityEstimation
from Models.utils import readPoiCoos, readTestData, readCategoryData, readTrainingData, readFriendData, saveModel, loadModel


class GeoSoCaMain:
    def main(datasetFiles, parameters):
        print("Started processing in GeoSoCa model ...")
        # Reading data from selected dataset
        numberOfUsers, numberOfPoI, numberOfCategories = open(datasetFiles['dataSize'], 'r').readlines()[
            0].strip('\n').split()
        numberOfUsers, numberOfPoI, numberOfCategories = int(
            numberOfUsers), int(numberOfPoI), int(numberOfCategories)
        usersList = list(range(numberOfUsers))
        poiList = list(range(numberOfPoI))
        np.random.shuffle(usersList)
        # Init values
        modelName = 'GeoSoCa'
        topK = parameters['topK']
        datasetName = parameters['datasetName']
        topRestricted = parameters['topRestricted']
        alpha = 0.5
        precision, recall = [], []
        # Load libraries
        AKDE = AdaptiveKernelDensityEstimation(alpha)
        SC = SocialCorrelation()
        CC = CategoricalCorrelation()
        print("Reading dataset instances ...")
        # Loading trainin items
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
        # Add caching policy (prevent a similar setting to be executed again) ---> Read from config
        executionRecord = open(
            f"./Generated/GeoSoCa_{datasetName}_top" + str(topRestricted) + ".txt", 'w+')
        # Calculating
        print("Evaluating results ...")
        for cnt, uid in enumerate(usersList):
            if uid in groundTruth:
                overallScores = [AKDE.predict(uid, lid) * SC.predict(uid, lid) * CC.predict(uid, lid)
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
