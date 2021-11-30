import numpy as np
from utils import logger
from config import LoreDict
from Evaluations.evaluator import evaluator
from Data.readDataSizes import readDataSizes
from Models.LORE.lib.FriendBasedCF import FriendBasedCF
from Models.LORE.lib.AdditiveMarkovChain import AdditiveMarkovChain
from Models.LORE.lib.KernelDensityEstimation import KernelDensityEstimation
from Models.utils import readFriendData, readPoiCoos, readSparseTrainingData, readTestData, readTrainingCheckins, saveModel, loadModel

modelName = 'LORE'


class LOREMain:
    def main(datasetFiles, parameters):
        logger(f'Processing data using {modelName} model ...')

        # Initializing model parameters
        alpha, deltaT = LoreDict['alpha'], LoreDict['deltaT']
        fusion, datasetName, evaluation = parameters[
            'fusion'], parameters['datasetName'], parameters['evaluation']

        # Reading data size from the selected dataset
        dataDictionary = readDataSizes(datasetName, datasetFiles)
        users, pois = dataDictionary['users'], dataDictionary['pois']

        # Computing the final scores
        print('Creating model-related variables ...')
        FCFScores, KDEScores, AMCScores = np.zeros((users['count'], pois['count'])), np.zeros(
            (users['count'], pois['count'])), np.zeros((users['count'], pois['count']))
        FCF = FriendBasedCF()
        KDE = KernelDensityEstimation()
        AMC = AdditiveMarkovChain(deltaT, alpha)
        # Loading trainin items
        logger('Reading dataset instances ...')
        sparseTrainingMatrix, trainingMatrix = readSparseTrainingData(
            datasetFiles['train'], users['count'], pois['count'])
        # Loading a sorted list of check-ins
        trainingCheckins = readTrainingCheckins(
            datasetFiles['checkins'], sparseTrainingMatrix)
        sortedTrainingCheckins = {uid: sorted(trainingCheckins[uid], key=lambda k: k[1])
                                  for uid in trainingCheckins}
        # Reading social data
        socialRelations = readFriendData(
            datasetFiles['socialRelations'], 'list', None)
        # Reading Ground-truth data
        groundTruth = readTestData(datasetFiles['test'])
        # Reading PoI data
        poiCoos = readPoiCoos(datasetFiles['poiCoos'])
        # Computations
        FCF.friendsSimilarityCalculation(
            socialRelations, poiCoos, sparseTrainingMatrix)
        KDE.precomputeKernelParameters(sparseTrainingMatrix, poiCoos)
        AMC.buildLocationToLocationTransitionGraph(sortedTrainingCheckins)
        # Processing items
        logger('Preparing Friend-based CF matrix ...')
        loadedModel = loadModel(modelName, datasetName,
                                f'FCF')
        if loadedModel == []:  # It should be created
            for cnt, uid in enumerate(users['list']):
                if uid in groundTruth:
                    for lid in pois['list']:
                        FCFScores[uid, lid] = FCF.predict(uid, lid)
            saveModel(FCFScores, modelName, datasetName,
                      f'FCF')
        else:  # It should be loaded
            FCFScores = loadedModel
        logger('Preparing Kernel Density Estimation matrix ...')
        loadedModel = loadModel(modelName, datasetName,
                                f'KDE')
        if loadedModel == []:  # It should be created
            for cnt, uid in enumerate(users['list']):
                if uid in groundTruth:
                    for lid in pois['list']:
                        KDEScores[uid, lid] = KDE.predict(uid, lid)
            saveModel(KDEScores, modelName, datasetName,
                      f'KDE')
        else:  # It should be loaded
            KDEScores = loadedModel
        logger('Preparing Additive Markov Chain matrix ...')
        loadedModel = loadModel(modelName, datasetName,
                                f'AMC')
        if loadedModel == []:  # It should be created
            for cnt, uid in enumerate(users['list']):
                if uid in groundTruth:
                    for lid in pois['list']:
                        AMCScores[uid, lid] = AMC.predict(uid, lid)
            saveModel(AMCScores, modelName, datasetName,
                      f'AMC')
        else:  # It should be loaded
            AMCScores = loadedModel
        # Evaluation
        evalParams = {'usersList': users['list'],
                      'groundTruth': groundTruth, 'fusion': fusion, 'poiList': pois['list'], 'trainingMatrix': trainingMatrix, 'evaluation': evaluation}
        modelParams = {'FCF': FCFScores, 'KDE': KDEScores, 'AMC': AMCScores}
        evaluator(modelName, datasetName, evalParams, modelParams)
