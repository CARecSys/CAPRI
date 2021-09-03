import time
import numpy as np


class CategoricalCorrelation(object):
    def __init__(self):
        self.Y = None
        self.gamma = None

    def loadModel(self, loadedModel):
        self.Y = loadedModel
        self.gamma = 1.0 + 1.0 / np.mean(np.log(1.0 + self.Y[self.Y > 0]))

    def computeGamma(self, checkinMatrix, poiCateMatrix):
        startTime = time.time()
        print("Precomputing categorical correlation parameter beta...", )
        B = checkinMatrix.dot(poiCateMatrix)
        P = poiCateMatrix.T
        Y = B.dot(P)
        gamma = 1.0 + 1.0 / np.mean(np.log(1.0 + Y[Y > 0]))
        elapsedTime = time.time() - startTime
        print("Finished in", '{:.2f}'.format(elapsedTime), "seconds.")
        self.gamma = gamma
        self.Y = Y

    def predict(self, u, l):
        return 1.0 - (1.0 + self.Y[u, l]) ** (1 - self.gamma)
