import numpy as np
from Evaluations.metricsAccuracy import precisionk, recallk
from Models.LORE.lib.FriendBasedCF import FriendBasedCF
from Models.LORE.lib.AdditiveMarkovChain import AdditiveMarkovChain
from Models.LORE.lib.KernelDensityEstimation import KernelDensityEstimation
from Models.utils import readFriendData, readPoiCoos, readSparseTrainingData, readTestData, readTrainingCheckins


class LOREMain:
    def main(datasetFiles, parameters):
        print("Started processing in LORE model ...")
        # Reading data from selected dataset
        numberOfUsers, numberOfPoI = open(datasetFiles['dataSize'], 'r').readlines()[
            0].strip('\n').split()
        numberOfUsers, numberOfPoI = int(numberOfUsers), int(numberOfPoI)
        usersList = list(range(numberOfUsers))
        poiList = list(range(numberOfPoI))
        np.random.shuffle(usersList)
        # Init values
        alpha = 0.05
        deltaT = 3600 * 24
        modelName = 'LORE'
        precision, recall = [], []
        topK = parameters['topK']
        datasetName = parameters['datasetName']
        topRestricted = parameters['topRestricted']
        # Load libraries
        FCF = FriendBasedCF()
        KDE = KernelDensityEstimation()
        AMC = AdditiveMarkovChain(deltaT, alpha)
        print("Reading dataset instances ...")
        # Loading trainin items
        sparseTrainingMatrix, trainingTuples = readSparseTrainingData(
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
        # Calculating
        print("Evaluating results ...")
        for cnt, uid in enumerate(usersList):
            if uid in groundTruth:
                overallScores = [KDE.predict(uid, lid) * FCF.predict(uid, lid) * AMC.predict(uid, lid)
                                 if (uid, lid) not in trainingTuples else -1
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
