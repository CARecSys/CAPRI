import numpy as np
from utils import logger
from config import sparsityRatio
from Evaluations.evaluator import evaluator
from Models.USG.lib.PowerLaw import PowerLaw
from Models.USG.lib.UserBasedCF import UserBasedCF
from Models.USG.lib.FriendBasedCF import FriendBasedCF
from Models.utils import readTrainingData, readFriendData, readTestData, readPoiCoos, saveModel, loadModel

modelName = 'USG'


class USGMain:
    def main(datasetFiles, parameters):
        logger(f'Started processing data using {modelName} ...')
        # Reading data from the selected dataset
        numberOfUsers, numberOfPoI = open(datasetFiles['dataSize'], 'r').readlines()[
            0].strip('\n').split()
        numberOfUsers, numberOfPoI = int(numberOfUsers), int(numberOfPoI)
        usersList = list(range(numberOfUsers))
        poiList = list(range(numberOfPoI))
        np.random.shuffle(usersList)
        # Init values
        fusion, datasetName, evaluation = parameters[
            'fusion'], parameters['datasetName'], parameters['evaluation']
        UScores = np.zeros((numberOfUsers, numberOfPoI))
        SScores = np.zeros((numberOfUsers, numberOfPoI))
        GScores = np.zeros((numberOfUsers, numberOfPoI))
        # Load libraries
        U = UserBasedCF()
        S = FriendBasedCF(eta=0.05)
        G = PowerLaw()
        logger('Reading dataset instances ...')
        # Loading training items
        trainingMatrix = readTrainingData(
            datasetFiles['train'], numberOfUsers, numberOfPoI, False)
        # Reading Ground-truth data
        groundTruth = readTestData(datasetFiles['test'])
        # Reading social data
        socialRelations = readFriendData(
            datasetFiles['socialRelations'], 'dictionary', None)
        # Reading PoI data
        poiCoos = readPoiCoos(datasetFiles['poiCoos'])
        # User-based Collaborative Filtering Calculations
        loadedModel = loadModel(modelName, datasetName, 'recScore')
        if loadedModel == []:  # It should be created
            U.preComputeRecScores(trainingMatrix)
            saveModel(U.recScore, modelName, datasetName, 'recScore')
        else:  # It should be loaded
            U.loadModel(loadedModel)
        S.friendsSimilarityCalculation(socialRelations, trainingMatrix)
        G.fitDistanceDistribution(trainingMatrix, poiCoos)
        # Processing items
        logger('Preparing User-based CF matrix ...')
        loadedModel = loadModel(modelName, datasetName,
                                f'U_{sparsityRatio}')
        if loadedModel == []:  # It should be created
            for cnt, uid in enumerate(usersList):
                if uid in groundTruth:
                    for lid in poiList:
                        UScores[uid, lid] = U.predict(uid, lid)
                    UScores = np.array(UScores)
            saveModel(UScores, modelName, datasetName,
                      f'U_{sparsityRatio}')
        else:  # It should be loaded
            UScores = loadedModel
        logger('Preparing Friend-based CF matrix ...')
        loadedModel = loadModel(modelName, datasetName,
                                f'S_{sparsityRatio}')
        if loadedModel == []:  # It should be created
            for cnt, uid in enumerate(usersList):
                if uid in groundTruth:
                    for lid in poiList:
                        SScores[uid, lid] = S.predict(uid, lid)
                    SScores = np.array(SScores)
            saveModel(SScores, modelName, datasetName,
                      f'S_{sparsityRatio}')
        else:  # It should be loaded
            SScores = loadedModel
        logger('Preparing Power Law matrix ...')
        loadedModel = loadModel(modelName, datasetName,
                                f'G_{sparsityRatio}')
        if loadedModel == []:  # It should be created
            for cnt, uid in enumerate(usersList):
                if uid in groundTruth:
                    for lid in poiList:
                        GScores[uid, lid] = G.predict(uid, lid)
                    GScores = np.array(GScores)
            saveModel(GScores, modelName, datasetName,
                      f'G_{sparsityRatio}')
        else:  # It should be loaded
            GScores = loadedModel
        # Evaluation
        evalParams = {'usersList': usersList,
                      'groundTruth': groundTruth, 'fusion': fusion, 'poiList': poiList, 'trainingMatrix': trainingMatrix, 'evaluation': evaluation}
        modelParams = {'U': UScores, 'S': SScores, 'G': GScores}
        evaluator(modelName, datasetName, evalParams, modelParams)
