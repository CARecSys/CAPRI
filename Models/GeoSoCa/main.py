import numpy as np
from utils import logger
from Evaluations.evaluator import evaluator
from Data.readDataSizes import readDataSizes
from config import sparsityRatio, GeoSoCaDict
from Models.GeoSoCa.lib.SocialCorrelation import SocialCorrelation
from Models.GeoSoCa.lib.CategoricalCorrelation import CategoricalCorrelation
from Models.GeoSoCa.lib.AdaptiveKernelDensityEstimation import AdaptiveKernelDensityEstimation
from Models.utils import readPoiCoos, readTestData, readCategoryData, readTrainingData, readFriendData, saveModel, loadModel

modelName = 'GeoSoCa'


class GeoSoCaMain:
    def main(datasetFiles, parameters):
        logger(f'Started processing data using {modelName} ...')
        # Initializing model parameters
        alpha = GeoSoCaDict['alpha']
        fusion, datasetName, evaluation = parameters[
            'fusion'], parameters['datasetName'], parameters['evaluation']
        # Reading data size from the selected dataset
        dataDictionary = readDataSizes(datasetName, datasetFiles)
        users, pois, categories = dataDictionary['users'], dataDictionary['pois'], dataDictionary['categories']
        # Creating model-related libraries
        SCScores, CCScores, AKDEScores = np.zeros((users['count'], pois['count'])), np.zeros(
            (users['count'], pois['count'])), np.zeros((users['count'], pois['count']))
        AKDE = AdaptiveKernelDensityEstimation(alpha)
        SC = SocialCorrelation()
        CC = CategoricalCorrelation()
        logger('Reading dataset instances ...')
        # Loading training items
        trainingMatrix = readTrainingData(
            datasetFiles['train'], users['count'], pois['count'], True)
        socialRelations = readFriendData(
            datasetFiles['socialRelations'], 'ndarray', users['count'])
        groundTruth = readTestData(datasetFiles['test'])
        poiCoos = readPoiCoos(datasetFiles['poiCoos'])
        poiCategoryMatrix = readCategoryData(
            datasetFiles['poiCategories'], categories['count'], pois['count'])
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
            for cnt, uid in enumerate(users['list']):
                if uid in groundTruth:
                    for lid in pois['list']:
                        AKDEScores[uid, lid] = AKDE.predict(uid, lid)
            saveModel(AKDEScores, modelName, datasetName,
                      f'AKDE_{sparsityRatio}')
        else:  # It should be loaded
            AKDEScores = loadedModel
        logger('Preparing Social Correlation matrix ...')
        loadedModel = loadModel(modelName, datasetName,
                                f'SC_{sparsityRatio}')
        if loadedModel == []:  # It should be created
            for cnt, uid in enumerate(users['list']):
                if uid in groundTruth:
                    for lid in pois['list']:
                        SCScores[uid, lid] = SC.predict(uid, lid)
            saveModel(SCScores, modelName, datasetName,
                      f'SC_{sparsityRatio}')
        else:  # It should be loaded
            SCScores = loadedModel
        logger('Preparing Categorical Correlation matrix ...')
        loadedModel = loadModel(modelName, datasetName,
                                f'CC_{sparsityRatio}')
        if loadedModel == []:  # It should be created
            for cnt, uid in enumerate(users['list']):
                if uid in groundTruth:
                    for lid in pois['list']:
                        CCScores[uid, lid] = CC.predict(uid, lid)
            saveModel(CCScores, modelName, datasetName,
                      f'CC_{sparsityRatio}')
        else:  # It should be loaded
            CCScores = loadedModel
        # Evaluation
        evalParams = {'usersList': users['list'],
                      'groundTruth': groundTruth, 'fusion': fusion, 'poiList': pois['list'], 'trainingMatrix': trainingMatrix, 'evaluation': evaluation}
        modelParams = {'AKDE': AKDEScores, 'SC': SCScores, 'CC': CCScores}
        evaluator(modelName, datasetName, evalParams, modelParams)
