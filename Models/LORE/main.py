import numpy as np
from utils import logger, textToOperator
from Models.LORE.lib.FriendBasedCF import FriendBasedCF
from Evaluations.metrics.accuracy import precisionk, recallk
from config import topK, sparsityRatio, topRestricted, LoreDict
from Models.LORE.lib.AdditiveMarkovChain import AdditiveMarkovChain
from Models.LORE.lib.KernelDensityEstimation import KernelDensityEstimation
from Models.utils import readFriendData, readPoiCoos, readSparseTrainingData, readTestData, readTrainingCheckins, saveModel, loadModel


class LOREMain:
    def main(datasetFiles, parameters):
        logger('Started processing in LORE model ...')
        # Reading data from selected dataset
        numberOfUsers, numberOfPoI = open(datasetFiles['dataSize'], 'r').readlines()[
            0].strip('\n').split()
        numberOfUsers, numberOfPoI = int(numberOfUsers), int(numberOfPoI)
        usersList = list(range(numberOfUsers))
        poiList = list(range(numberOfPoI))
        np.random.shuffle(usersList)
        # Init values
        modelName = 'LORE'
        alpha, deltaT = LoreDict['alpha'], LoreDict['deltaT']
        precision, recall = [], []
        fusion = parameters['fusion']
        datasetName = parameters['datasetName']
        FCFScores = np.zeros((numberOfUsers, numberOfPoI))
        KDEScores = np.zeros((numberOfUsers, numberOfPoI))
        AMCScores = np.zeros((numberOfUsers, numberOfPoI))
        # Load libraries
        FCF = FriendBasedCF()
        KDE = KernelDensityEstimation()
        AMC = AdditiveMarkovChain(deltaT, alpha)
        logger('Reading dataset instances ...')
        # Loading trainin items
        sparseTrainingMatrix, trainingMatrix = readSparseTrainingData(
            datasetFiles['train'], numberOfUsers, numberOfPoI)
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
        # Add caching policy (prevent a similar setting to be executed again)
        executionRecord = open(
            f"./Generated/LORE_{datasetName}_top" + str(topRestricted) + ".txt", 'w+')
        # Processing items
        logger('Preparing Friend-based CF matrix ...')
        loadedModel = loadModel(modelName, datasetName,
                                f'FCF_{sparsityRatio}')
        if loadedModel == []:  # It should be created
            for cnt, uid in enumerate(usersList):
                if uid in groundTruth:
                    for lid in poiList:
                        FCFScores[uid, lid] = FCF.predict(uid, lid)
            saveModel(FCFScores, modelName, datasetName,
                      f'FCF_{sparsityRatio}')
        else:  # It should be loaded
            FCFScores = loadedModel
        logger('Preparing Kernel Density Estimation matrix ...')
        loadedModel = loadModel(modelName, datasetName,
                                f'KDE_{sparsityRatio}')
        if loadedModel == []:  # It should be created
            for cnt, uid in enumerate(usersList):
                if uid in groundTruth:
                    for lid in poiList:
                        KDEScores[uid, lid] = KDE.predict(uid, lid)
            saveModel(KDEScores, modelName, datasetName,
                      f'KDE_{sparsityRatio}')
        else:  # It should be loaded
            KDEScores = loadedModel
        logger('Preparing Additive Markov Chain matrix ...')
        loadedModel = loadModel(modelName, datasetName,
                                f'AMC_{sparsityRatio}')
        if loadedModel == []:  # It should be created
            for cnt, uid in enumerate(usersList):
                if uid in groundTruth:
                    for lid in poiList:
                        AMCScores[uid, lid] = AMC.predict(uid, lid)
            saveModel(AMCScores, modelName, datasetName,
                      f'AMC_{sparsityRatio}')
        else:  # It should be loaded
            AMCScores = loadedModel
        # Calculating
        logger('Evaluating results ...')
        for cnt, uid in enumerate(usersList):
            if uid in groundTruth:
                overallScores = [textToOperator(fusion, [KDEScores[uid, lid], FCFScores[uid, lid], AMCScores[uid, lid]])
                                 if (uid, lid) not in trainingMatrix else -1
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
