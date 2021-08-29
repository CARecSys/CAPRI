import numpy as np
from Evaluations.metrics import precisionk, recallk
from Models.LORE.lib.FriendBasedCF import FriendBasedCF
from Models.LORE.lib.AdditiveMarkovChain import AdditiveMarkovChain
from Models.LORE.lib.KernelDensityEstimation import KernelDensityEstimation
from Models.utils import readFriendData, readPoiCoos, readSparseTrainingData, readTestData, readTrainingCheckins


class LOREMain:
    def main(datasetFiles, selectedDataset):
        print("Started processing in LORE model ...")
        # Reading data from selected dataset
        numberOfUsers, numberOfPoI = open(datasetFiles['dataSize'], 'r').readlines()[
            0].strip('\n').split()
        numberOfUsers, numberOfPoI = int(numberOfUsers), int(numberOfPoI)
        usersList = list(range(numberOfUsers))
        poiList = list(range(numberOfPoI))
        np.random.shuffle(usersList)
        # Init values
        topK = 100
        alpha = 0.05
        deltaT = 3600 * 24
        precision, recall = [], []
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
        socialRelations = readFriendData(datasetFiles['socialRelations'])
        # Reading Ground-truth data
        groundTruth = readTestData(datasetFiles['test'])
        # Reading PoI data
        poiCoos = readPoiCoos(datasetFiles['poiCoos'])
        # Computations
        FCF.compute_friend_sim(socialRelations, poiCoos, sparseTrainingMatrix)
        KDE.precompute_kernel_parameters(sparseTrainingMatrix, poiCoos)
        AMC.build_location_location_transition_graph(sortedTrainingCheckins)
        # Add caching policy (prevent a similar setting to be executed again) ---> Read from config
        executionRecord = open(
            f"./Generated/LORE_{selectedDataset}_top" + str(topK) + ".txt", 'w+')
        # Calculating
        print("Evaluating results ...")
        for cnt, uid in enumerate(usersList):
            if uid in groundTruth:
                overallScores = [KDE.predict(uid, lid) * FCF.predict(uid, lid) * AMC.predict(uid, lid)
                                 if (uid, lid) not in trainingTuples else -1
                                 for lid in poiList]
                overallScores = np.array(overallScores)
                predicted = list(reversed(overallScores.argsort()))[:topK]
                actual = groundTruth[uid]
                precision.append(precisionk(actual, predicted[:10]))
                recall.append(recallk(actual, predicted[:10]))
                print(cnt, uid, "pre@10:", np.mean(precision),
                      "rec@10:", np.mean(recall))
                executionRecord.write('\t'.join([
                    str(cnt),
                    str(uid),
                    ','.join([str(lid) for lid in predicted])
                ]) + '\n')
