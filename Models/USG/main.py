import numpy as np
from Models.USG.lib.PowerLaw import PowerLaw
from Models.USG.lib.UserBasedCF import UserBasedCF
from Evaluations.metrics import precisionk, recallk
from Models.USG.lib.FriendBasedCF import FriendBasedCF
from Models.utils import normalize, readTrainingData, readFriendData, readTestData, readPoiCoos


class USGMain:
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
        alpha = 0.1
        beta = 0.1
        precision, recall = [], []
        # Load libraries
        U = UserBasedCF()
        S = FriendBasedCF(eta=0.05)
        G = PowerLaw()
        print("Reading dataset instances ...")
        # Loading training items
        trainingMatrix = readTrainingData(
            datasetFiles['train'], numberOfUsers, numberOfPoI)
        # Reading Ground-truth data
        groundTruth = readTestData(datasetFiles['test'])
        # Reading social data
        socialRelations = readFriendData(datasetFiles['socialRelations'])
        # Reading PoI data
        poiCoos = readPoiCoos(datasetFiles['poiCoos'])
        # Computations
        U.pre_compute_rec_scores(trainingMatrix)
        S.compute_friend_sim(socialRelations, trainingMatrix)
        G.fit_distance_distribution(trainingMatrix, poiCoos)
        # Add caching policy (prevent a similar setting to be executed again) ---> Read from config
        executionRecord = open(
            f"./Generated/USG_{selectedDataset}_top" + str(topK) + ".txt", 'w+')
        # Calculating
        print("Evaluating results ...")
        for cnt, uid in enumerate(usersList):
            if uid in groundTruth:
                U_scores = normalize([U.predict(uid, lid)
                                      if trainingMatrix[uid, lid] == 0 else -1
                                      for lid in poiList])
                S_scores = normalize([S.predict(uid, lid)
                                      if trainingMatrix[uid, lid] == 0 else -1
                                      for lid in poiList])
                G_scores = normalize([G.predict(uid, lid)
                                      if trainingMatrix[uid, lid] == 0 else -1
                                      for lid in poiList])
                U_scores = np.array(U_scores)
                S_scores = np.array(S_scores)
                G_scores = np.array(G_scores)
                overallScores = (1.0 - alpha - beta) * U_scores + \
                    alpha * S_scores + beta * G_scores
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
