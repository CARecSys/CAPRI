import os
import time
import numpy as np
import scipy.sparse as sparse
from collections import defaultdict


def readSparseTrainingData(trainFile, numberOfUsers, numberOfPoI):
    trainingData = open(trainFile, 'r').readlines()
    sparseTrainingMatrix = sparse.dok_matrix((numberOfUsers, numberOfPoI))
    trainingTuples = set()
    for dataInstance in trainingData:
        uid, lid, freq = dataInstance.strip().split()
        uid, lid, freq = int(uid), int(lid), int(freq)
        sparseTrainingMatrix[uid, lid] = freq
        trainingTuples.add((uid, lid))
    return sparseTrainingMatrix, trainingTuples


# withFrequency: True for GeoSoCa, False for USG
def readTrainingData(trainFile, numberOfUsers, numberOfPoI, withFrequency):
    trainingData = open(trainFile, 'r').readlines()
    trainingMatrix = np.zeros((numberOfUsers, numberOfPoI))
    # TODO: we may replace this condition with a more compact one
    # e.g. value = freq if withFrequency == True else 1.0
    if withFrequency == True:
        for dataInstance in trainingData:
            uid, lid, freq = dataInstance.strip().split()
            uid, lid, freq = int(uid), int(lid), int(freq)
            trainingMatrix[uid, lid] = freq
    else:
        for dataInstance in trainingData:
            uid, lid, _ = dataInstance.strip().split()
            uid, lid = int(uid), int(lid)
            trainingMatrix[uid, lid] = 1.0
    return trainingMatrix


def readTrainingCheckins(checkinFile, sparseTrainingMatrix):
    checkinData = open(checkinFile, 'r').readlines()
    trainingCheckins = defaultdict(list)
    for dataInstance in checkinData:
        uid, lid, ctime = dataInstance.strip().split()
        uid, lid, ctime = int(uid), int(lid), float(ctime)
        if not sparseTrainingMatrix[uid, lid] == 0:
            trainingCheckins[uid].append([lid, ctime])
    return trainingCheckins


# appendType: 'list' for GeoSoCa and USG, 2) 'dictionary' for USG
def readFriendData(socialFile, appendType, numberOfUsers):
    socialData = open(socialFile, 'r').readlines()
    # TODO: we may replace this condition with a more compact one
    if appendType == 'list':  # LORE, GeoSoCa
        # GeoSoCa needs numberOfUsers, but LORE doesn't
        socialRelations = [] if numberOfUsers == None else np.zeros(
            (numberOfUsers, numberOfUsers))
        for dataInstance in socialData:
            uid1, uid2 = dataInstance.strip().split()
            uid1, uid2 = int(uid1), int(uid2)
            socialRelations.append([uid1, uid2])
        return socialRelations
    else:  # USG
        socialRelations = defaultdict(list)
        for dataInstance in socialData:
            uid1, uid2 = dataInstance.strip().split()
            uid1, uid2 = int(uid1), int(uid2)
            socialRelations[uid1].append(uid2)
            socialRelations[uid2].append(uid1)
        return socialRelations


def readTestData(testFile):
    groundTruth = defaultdict(set)
    truthData = open(testFile, 'r').readlines()
    for dataInstance in truthData:
        uid, lid, _ = dataInstance.strip().split()
        uid, lid = int(uid), int(lid)
        groundTruth[uid].add(lid)
    return groundTruth


def readPoiCoos(poiFile):
    poiCoos = {}
    poiData = open(poiFile, 'r').readlines()
    for dataInstance in poiData:
        lid, lat, lng = dataInstance.strip().split()
        lid, lat, lng = int(lid), float(lat), float(lng)
        poiCoos[lid] = (lat, lng)
    return poiCoos


def readCategoryData(categoryFile, numberOfCategories, numberOfPoI):
    categoryData = open(categoryFile, 'r').readlines()
    poiCategoryMatrix = np.zeros((numberOfPoI, numberOfCategories))
    for dataInstance in categoryData:
        lid, cid = dataInstance.strip().split()
        lid, cid = int(lid), int(cid)
        poiCategoryMatrix[lid, cid] = 1.0
    return poiCategoryMatrix


def normalize(scores):
    maxScore = max(scores)
    if not maxScore == 0:
        scores = [s / maxScore for s in scores]
    return scores


def loadModel(modelName, datasetName, numberOfItems):
    startTime = time.time()
    print("Looking for model ...",)
    fileName = f'{modelName}_{datasetName}_{numberOfItems}'
    path = os.path.abspath(f'./{modelName}/savedModels/{fileName}.npy')
    content = np.load(path)
    print(content)
    elapsedTime = time.time() - startTime
    print(f"Loaded file in ",
          '{:.2f}'.format(elapsedTime), " seconds.")


def saveModel(content, modelName, datasetName, numberOfItems):
    startTime = time.time()
    print("Saving result...",)
    fileName = f'{modelName}_{datasetName}_{numberOfItems}'
    path = os.path.abspath(f'./{modelName}/savedModels/')
    np.save(path + fileName, content)
    elapsedTime = time.time() - startTime
    print(f"Saved file in ",
          '{:.2f}'.format(elapsedTime), " seconds ({path}/{fileName}.npy)")
