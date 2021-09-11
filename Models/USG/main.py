import numpy as np
from Models.USG.lib.PowerLaw import PowerLaw
from Models.USG.lib.UserBasedCF import UserBasedCF
from Evaluations.metricsAccuracy import precisionk, recallk
from Models.USG.lib.FriendBasedCF import FriendBasedCF
from Models.utils import normalize, readTrainingData, readFriendData, readTestData, readPoiCoos, saveModel, loadModel


class USGMain:
    def main(datasetFiles, parameters):
        print("Started processing in USG model ...")
        # Reading data from selected dataset
        numberOfUsers, numberOfPoI = open(datasetFiles['dataSize'], 'r').readlines()[
            0].strip('\n').split()
        numberOfUsers, numberOfPoI = int(numberOfUsers), int(numberOfPoI)
        usersList = list(range(numberOfUsers))
        poiList = list(range(numberOfPoI))
        np.random.shuffle(usersList)
        # Init values
        beta = 0.1
        alpha = 0.1
        modelName = 'USG'
        precision, recall = [], []
        topK = parameters['topK']
        datasetName = parameters['datasetName']
        topRestricted = parameters['topRestricted']
        # Load libraries
        U = UserBasedCF()
        S = FriendBasedCF(eta=0.05)
        G = PowerLaw()
        print("Reading dataset instances ...")
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
        # Add caching policy (prevent a similar setting to be executed again)
        executionRecord = open(
            f"./Generated/USG_{datasetName}_top" + str(topRestricted) + ".txt", 'w+')
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
