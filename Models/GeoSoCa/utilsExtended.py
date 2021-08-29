import numpy as np


def readFriendData(socialFile, numberOfUsers):
    socialData = open(socialFile, 'r').readlines()
    socialRelations = np.zeros((numberOfUsers, numberOfUsers))
    for dataInstance in socialData:
        uid1, uid2 = dataInstance.strip().split()
        uid1, uid2 = int(uid1), int(uid2)
        socialRelations[uid1, uid2] = 1.0
        socialRelations[uid2, uid1] = 1.0
    return socialRelations


def readTrainingData(trainFile, numberOfUsers, numberOfPoI):
    trainingData = open(trainFile, 'r').readlines()
    trainingMatrix = np.zeros((numberOfUsers, numberOfPoI))
    for dataInstance in trainingData:
        uid, lid, freq = dataInstance.strip().split()
        uid, lid, freq = int(uid), int(lid), int(freq)
        trainingMatrix[uid, lid] = freq
    return trainingMatrix
