import numpy as np
from utils import logger
from Evaluations.evaluator import evaluator
from config import sparsityRatio, GeoSoCaDict
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
        modelName = 'GeoSoCa'
        alpha = GeoSoCaDict['alpha']
        fusion, datasetName, evaluation = parameters[
            'fusion'], parameters['datasetName'], parameters['evaluation']
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
        # Evaluation
        evalParams = {'usersList': usersList,
                      'groundTruth': groundTruth, 'fusion': fusion, 'poiList': poiList, 'trainingMatrix': trainingMatrix, 'evaluation': evaluation}
        modelParams = {'AKDE': AKDEScores, 'SC': SCScores, 'CC': CCScores}
        evaluator(modelName, datasetName, evalParams, modelParams)
