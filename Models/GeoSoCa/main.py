import numpy as np
from Evaluations.metrics import precisionk, recallk
from Models.utils import readPoiCoos, readTestData, readCategoryData
from Models.GeoSoCa.lib.SocialCorrelation import SocialCorrelation
from Models.GeoSoCa.utilsExtended import readFriendData, readTrainingData
from Models.GeoSoCa.lib.CategoricalCorrelation import CategoricalCorrelation
from Models.GeoSoCa.lib.AdaptiveKernelDensityEstimation import AdaptiveKernelDensityEstimation


class GeoSoCaMain:
    def main(datasetFiles, selectedDataset):
        print("Started processing in GeoSoCa model ...")
        # Reading data from selected dataset
        numberOfUsers, numberOfPoI, numberOfCategories = open(datasetFiles['dataSize'], 'r').readlines()[
            0].strip('\n').split()
        numberOfUsers, numberOfPoI, numberOfCategories = int(
            numberOfUsers), int(numberOfPoI), int(numberOfCategories)
        usersList = list(range(numberOfUsers))
        poiList = list(range(numberOfPoI))
        np.random.shuffle(usersList)
        # Init values
        topRestricted = 100
        alpha = 0.5
        precision, recall = [], []
        # Load libraries
        AKDE = AdaptiveKernelDensityEstimation(alpha)
        SC = SocialCorrelation()
        CC = CategoricalCorrelation()
        print("Reading dataset instances ...")
        # Loading trainin items
        trainingMatrix = readTrainingData(
            datasetFiles['train'], numberOfUsers, numberOfPoI)
        socialRelations = readFriendData(
            datasetFiles['socialRelations'], numberOfUsers)
        groundTruth = readTestData(datasetFiles['test'])
        poiCoos = readPoiCoos(datasetFiles['poiCoos'])
        poiCategoryMatrix = readCategoryData(
            datasetFiles['poiCategories'], numberOfCategories, numberOfPoI)
        # Computations
        AKDE.precomputeKernelParameters(trainingMatrix, poiCoos)
        SC.compute_beta(trainingMatrix, socialRelations)
        # SC.save_result("../savedModels/")
        CC.compute_gamma(trainingMatrix, poiCategoryMatrix)
        # CC.save_result("../savedModels/")
        # Add caching policy (prevent a similar setting to be executed again) ---> Read from config
        executionRecord = open(
            f"./Generated/GeoSoCa_{selectedDataset}_top" + str(topRestricted) + ".txt", 'w+')
        # Calculating
        print("Evaluating results ...")
        for cnt, uid in enumerate(usersList):
            if uid in groundTruth:
                overallScores = [AKDE.predict(uid, lid) * SC.predict(uid, lid) * CC.predict(uid, lid)
                                 if trainingMatrix[uid, lid] == 0 else -1
                                 for lid in poiList]
                overallScores = np.array(overallScores)
                predicted = list(reversed(overallScores.argsort()))[
                    :topRestricted]
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
