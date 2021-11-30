import numpy as np
from utils import logger
from config import USGDict
from Evaluations.evaluator import evaluator
from Data.readDataSizes import readDataSizes
from Models.USG.lib.PowerLaw import PowerLaw
from Models.USG.lib.UserBasedCF import UserBasedCF
from Models.USG.lib.FriendBasedCF import FriendBasedCF
from Models.utils import readTrainingData, readFriendData, readTestData, readPoiCoos, saveModel, loadModel

modelName = 'USG'


class USGMain:
    def main(datasetFiles, parameters):
        logger(f'Processing data using {modelName} model ...')
        # Initializing model parameters
        fusion, datasetName, evaluation = parameters[
            'fusion'], parameters['datasetName'], parameters['evaluation']
        # Reading data size from the selected dataset
        dataDictionary = readDataSizes(datasetName, datasetFiles)
        users, pois = dataDictionary['users'], dataDictionary['pois']
        # Creating model-related libraries
        print('Creating model-related variables ...')
        UScores, SScores, GScores = np.zeros((users['count'], pois['count'])), np.zeros(
            (users['count'], pois['count'])), np.zeros((users['count'], pois['count']))
        U = UserBasedCF()
        S = FriendBasedCF(USGDict['eta'])
        G = PowerLaw()
        # Loading training items
        logger('Reading dataset instances ...')
        trainingMatrix = readTrainingData(
            datasetFiles['train'], users['count'], pois['count'], False)
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
                                f'U')
        if loadedModel == []:  # It should be created
            for cnt, uid in enumerate(users['list']):
                if uid in groundTruth:
                    for lid in pois['list']:
                        UScores[uid, lid] = U.predict(uid, lid)
                    UScores = np.array(UScores)
            saveModel(UScores, modelName, datasetName,
                      f'U')
        else:  # It should be loaded
            UScores = loadedModel
        logger('Preparing Friend-based CF matrix ...')
        loadedModel = loadModel(modelName, datasetName,
                                f'S')
        if loadedModel == []:  # It should be created
            for cnt, uid in enumerate(users['list']):
                if uid in groundTruth:
                    for lid in pois['list']:
                        SScores[uid, lid] = S.predict(uid, lid)
                    SScores = np.array(SScores)
            saveModel(SScores, modelName, datasetName,
                      f'S')
        else:  # It should be loaded
            SScores = loadedModel
        logger('Preparing Power Law matrix ...')
        loadedModel = loadModel(modelName, datasetName,
                                f'G')
        if loadedModel == []:  # It should be created
            for cnt, uid in enumerate(users['list']):
                if uid in groundTruth:
                    for lid in pois['list']:
                        GScores[uid, lid] = G.predict(uid, lid)
                    GScores = np.array(GScores)
            saveModel(GScores, modelName, datasetName,
                      f'G')
        else:  # It should be loaded
            GScores = loadedModel
        # Evaluation
        evalParams = {'usersList': users['list'],
                      'groundTruth': groundTruth, 'fusion': fusion, 'poiList': pois['list'], 'trainingMatrix': trainingMatrix, 'evaluation': evaluation}
        modelParams = {'U': UScores, 'S': SScores, 'G': GScores}
        evaluator(modelName, datasetName, evalParams, modelParams)
