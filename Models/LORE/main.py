from Models.LORE.utils import readFriendData, readPoiCoos, readSparseTrainingData, readTestData, readTrainingCheckins
from Models.LORE.lib.AdditiveMarkovChain import AdditiveMarkovChain
from Models.LORE.lib.KernelDensityEstimation import KernelDensityEstimation
from Models.LORE.lib.FriendBasedCF import FriendBasedCF


class LOREMain:
    def main(datasetFiles):
        print("Started processing in LORE model ...")
        # Reading data from selected dataset
        numberOfUsers, numberOfPoI = open(datasetFiles['dataSize'], 'r').readlines()[
            0].strip('\n').split()
        numberOfUsers, numberOfPoI = int(numberOfUsers), int(numberOfPoI)
        # Init values
        topK = 100
        deltaT = 3600 * 24
        alpha = 0.05
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
