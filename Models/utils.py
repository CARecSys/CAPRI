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


def readTrainingData(trainFile, numberOfUsers, numberOfPoI):
    trainingData = open(trainFile, 'r').readlines()
    trainingMatrix = np.zeros((numberOfUsers, numberOfPoI))
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


def readFriendData(socialFile):
    socialData = open(socialFile, 'r').readlines()
    socialRelations = defaultdict(list)
    for dataInstance in socialData:
        uid1, uid2 = dataInstance.strip().split()
        uid1, uid2 = int(uid1), int(uid2)
        socialRelations.append([uid1, uid2])
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


def normalize(scores):
    maxScore = max(scores)
    if not maxScore == 0:
        scores = [s / maxScore for s in scores]
    return scores
