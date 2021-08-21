import numpy as np
import scipy.sparse as sparse

from collections import defaultdict

from lib.FriendBasedCF import FriendBasedCF
from lib.KernelDensityEstimation import KernelDensityEstimation
from lib.AdditiveMarkovChain import AdditiveMarkovChain

from lib.metrics import precisionk, recallk


def read_friend_data():
    social_data = open(social_file, 'r').readlines()
    social_relations = []
    for eachline in social_data:
        uid1, uid2 = eachline.strip().split()
        uid1, uid2 = int(uid1), int(uid2)
        social_relations.append([uid1, uid2])
    return social_relations


def read_poi_coos():
    poi_coos = {}
    poi_data = open(poi_file, 'r').readlines()
    for eachline in poi_data:
        lid, lat, lng = eachline.strip().split()
        lid, lat, lng = int(lid), float(lat), float(lng)
        poi_coos[lid] = (lat, lng)
    return poi_coos


def read_training_check_ins(training_matrix):
    check_in_data = open(check_in_file, 'r').readlines()
    training_check_ins = defaultdict(list)
    for eachline in check_in_data:
        uid, lid, ctime = eachline.strip().split()
        uid, lid, ctime = int(uid), int(lid), float(ctime)
        if not training_matrix[uid, lid] == 0:
            training_check_ins[uid].append([lid, ctime])
    return training_check_ins


def read_sparse_training_data():
    train_data = open(train_file, 'r').readlines()
    sparse_training_matrix = sparse.dok_matrix((user_num, poi_num))
    training_tuples = set()
    for eachline in train_data:
        uid, lid, freq = eachline.strip().split()
        uid, lid, freq = int(uid), int(lid), int(freq)
        sparse_training_matrix[uid, lid] = freq
        training_tuples.add((uid, lid))
    return sparse_training_matrix, training_tuples


def read_ground_truth():
    ground_truth = defaultdict(set)
    truth_data = open(test_file, 'r').readlines()
    for eachline in truth_data:
        uid, lid, _ = eachline.strip().split()
        uid, lid = int(uid), int(lid)
        ground_truth[uid].add(lid)
    return ground_truth


def main():
    sparse_training_matrix, training_tuples = read_sparse_training_data()
    training_check_ins = read_training_check_ins(sparse_training_matrix)
    sorted_training_check_ins = {uid: sorted(training_check_ins[uid], key=lambda k: k[1])
                                 for uid in training_check_ins}
    social_relations = read_friend_data()
    ground_truth = read_ground_truth()
    poi_coos = read_poi_coos()

    FCF.compute_friend_sim(social_relations, poi_coos, sparse_training_matrix)
    KDE.precompute_kernel_parameters(sparse_training_matrix, poi_coos)
    AMC.build_location_location_transition_graph(sorted_training_check_ins)

    result_out = open("./result/gis14_top_" + str(top_k) + ".txt", 'w')

    all_uids = list(range(user_num))
    all_lids = list(range(poi_num))
    np.random.shuffle(all_uids)

    precision, recall = [], []
    for cnt, uid in enumerate(all_uids):
        if uid in ground_truth:
            overall_scores = [KDE.predict(uid, lid) * FCF.predict(uid, lid) * AMC.predict(uid, lid)
                              if (uid, lid) not in training_tuples else -1
                              for lid in all_lids]
            overall_scores = np.array(overall_scores)

            predicted = list(reversed(overall_scores.argsort()))[:top_k]
            actual = ground_truth[uid]

            precision.append(precisionk(actual, predicted[:10]))
            recall.append(recallk(actual, predicted[:10]))

            print(cnt, uid, "pre@10:", np.mean(precision), "rec@10:", np.mean(recall))
            result_out.write('\t'.join([
                str(cnt),
                str(uid),
                ','.join([str(lid) for lid in predicted])
            ]) + '\n')


if __name__ == '__main__':
    data_dir = "../data/"

    size_file = data_dir + "Gowalla_data_size.txt"
    check_in_file = data_dir + "Gowalla_checkins.txt"
    train_file = data_dir + "Gowalla_train.txt"
    tune_file = data_dir + "Gowalla_tune.txt"
    test_file = data_dir + "Gowalla_test.txt"
    social_file = data_dir + "Gowalla_social_relations.txt"
    poi_file = data_dir + "Gowalla_poi_coos.txt"

    user_num, poi_num = open(size_file, 'r').readlines()[0].strip('\n').split()
    user_num, poi_num = int(user_num), int(poi_num)

    top_k = 100

    FCF = FriendBasedCF()
    KDE = KernelDensityEstimation()
    AMC = AdditiveMarkovChain(delta_t=3600*24, alpha=0.05)

    main()
