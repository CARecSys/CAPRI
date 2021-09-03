import time
import numpy as np
from numpy.linalg import norm


class UserBasedCF(object):
    def __init__(self):
        self.recScore = None

    # def loadModel(self, path):
    #     startTime = time.time()
    #     print("Loading model ...",)
    #     self.recScore = np.load(path + "recScore.npy")
    #     print("Loaded in ", time.time() - startTime, "seconds")

    # def saveModel(self, path):
    #     startTime = time.time()
    #     print("Saving result...",)
    #     np.save(path + "recScore", self.recScore)
    #     print("Done. Elapsed time:", time.time() - startTime, "s")

    def preComputeRecScores(self, C):
        startTime = time.time()
        print("Training User-based Collaborative Filtering...", )
        sim = C.dot(C.T)
        norms = [norm(C[i]) for i in range(C.shape[0])]
        for i in range(C.shape[0]):
            sim[i][i] = 0.0
            for j in range(i+1, C.shape[0]):
                sim[i][j] /= (norms[i] * norms[j])
                sim[j][i] /= (norms[i] * norms[j])
        self.recScore = sim.dot(C)
        print("Finished in", time.time() - startTime, "seconds")

    def predict(self, i, j):
        return self.recScore[i][j]
