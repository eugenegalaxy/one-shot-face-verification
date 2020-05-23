#!/usr/bin/env python3

import sys
import os
import time
import logging
import datetime

from face_recognition.face_verification import FaceVerification, \
    g_THRESHOLD_UNCERTAINTY, g_WEIGHT_avg_dist, g_WEIGHT_min_dist, \
    g_KNN_or_SVC, g_TARGET_TO_DB_NAME_lowerSTD, g_TARGET_TO_DB_NAME_upperSTD, \
    g_UNRECOGNISED_RATIO, g_DMv3_lowerSTD, g_DMv3_upperSTD, g_DEFAULT_THRESHOLD


def accuracy(tp, tn, fp, fn):
    '''
    Accuracy is a ratio of correctly predicted observation to the total observations.
    '''
    Accuracy = (tp + tn) / (tp + fp + fn + tn)
    return Accuracy


def precision(tp, fp):
    '''
    Precision is the ratio of correctly predicted positive observations to the total predicted positive observations.
    The question that this metric answer is of all passengers that labeled as survived, how many actually survived?
    High precision relates to the low false positive rate.
    '''
    Precision = tp / (tp + fp)
    return Precision


def recall(tp, fn):
    '''
    Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes.
    The question recall answers is: Of all the passengers that truly survived, how many did we label?
    '''
    Recall = tp / (tp + fn)
    return Recall


def f1score(rec, prec):
    '''
    F1 Score is the weighted average of Precision and Recall.
    Usually more useful metric than accuracy.
    Best if the cost of false positives and false negatives is very different.
    '''
    F1_Score = 2 * (rec * prec) / (rec + prec)
    return F1_Score


def init_FaceRec(db_path):
    fv = FaceVerification()
    fv.initdb_and_classifier(db_path)

    ALL_FROM_DIRECTORY = 2
    fv.setImgMode(ALL_FROM_DIRECTORY)
    return fv


def predict_all(tg_path):
    all_db_names = [item.name for item in FV.db_metadata]
    recognised_names = []
    target_names = []
    min_dists = []
    for subdir in os.listdir(tg_path):
        if subdir == 'STRANGERS':
            pass
        else:
            target_names.append(subdir)
            name, min_dist = FV.Predict(directory_path=os.path.join(tg_path, subdir))
            recognised_names.append(name)
            min_dists.append(min_dist)
    return all_db_names, recognised_names, target_names, min_dists


def count_test(db_names, rec_names, tg_names, min_dists):
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    for idx, name in enumerate(rec_names):
        if name != 'Unrecognised':
            if name == tg_names[idx]:
                status = 'SUCCESS (True Positive)'
                tp += 1
            else:
                status = 'FAILED  (False Positive)'
                fp += 1
        else:
            if tg_names[idx] not in db_names:
                status = 'SUCCESS (True Negative)'
                tn += 1
            else:
                status = 'FAILED  (False Negative)'
                fn += 1

        print_str = ('{0} T: {1}\t({2:1.2f}) D: {3}\t-> {4}'.format(idx, tg_names[idx], min_dists[idx], name, status)).expandtabs(33)
        # logging.critical(print_str)
        # print(print_str)
    return fp, fn, tp, tn

time_start1 = time.time()
path = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(filename=path + '/test_log.log',
                    level=logging.CRITICAL,
                    # filemode='w',
                    format='%(message)s')

db1 = '/face_recognition/images/manual_database'  # Option 1
db2 = '/face_recognition/images/mysql_database'   # Option 2
chosen_database = db2
FV = init_FaceRec(path + chosen_database)
FV.plot_TSNE(FV.db_metadata, FV.db_features)
exit
print('ae')
tg1 = '/face_recognition/images/new_entries'
chosen_target = tg1

RUNS = 1

avg_acc = []
avg_prec = []
avg_rec = []
avg_f1 = []
avg_TP = []
avg_TN = []
avg_FP = []
avg_FN = []

now = datetime.datetime.now()
logging.critical("\n============================================================================================")
logging.critical("Session Time:")
logging.critical(now.strftime("%Y-%m-%d %H:%M:%S"))
logging.critical('Number of runs: {}'.format(RUNS))
print('Number of runs: {}'.format(RUNS))
logging.critical("Databse: {}.".format(chosen_database))
logging.critical("Targets: {}".format(chosen_target))

time2_start = time.time()
counter = 0
for x in range(RUNS):
    print('Run {}'.format(counter))
    db_names, recog_names, tg_names, min_distances = predict_all(path + chosen_target)

    FP, FN, TP, TN = count_test(db_names, recog_names, tg_names, min_distances)

    ACCURACY = accuracy(TP, TN, FP, FN)
    PRECISION = precision(TP, FP)
    RECALL = recall(TP, FN)
    F1SCORE = f1score(RECALL, PRECISION)

    avg_acc.append(ACCURACY)
    avg_prec.append(PRECISION)
    avg_rec.append(RECALL)
    avg_f1.append(F1SCORE)
    avg_TP.append(TP)
    avg_TN.append(TN)
    avg_FP.append(FP)
    avg_FN.append(FN)
    counter += 1

time2_stop = time.time()

a_acc = sum(avg_acc) / len(avg_acc)
a_prec = sum(avg_prec) / len(avg_prec)
a_rec = sum(avg_rec) / len(avg_rec)
a_f1 = sum(avg_f1) / len(avg_f1)
a_TP = sum(avg_TP) / len(avg_TP)
a_TN = sum(avg_TN) / len(avg_TN)
a_FP = sum(avg_FP) / len(avg_FP)
a_FN = sum(avg_FN) / len(avg_FN)

logging.critical('\nFalse Positive = {}'.format(int(a_FP)))
logging.critical('False Negative = {}'.format(int(a_FN)))
logging.critical('True Positive = {}'.format(int(a_TP)))
logging.critical('True Negative = {}'.format(int(a_TN)))

print('\nFalse Positive = {}'.format(int(a_FP)))
print('False Negative = {}'.format(int(a_FN)))
print('True Positive = {}'.format(int(a_TP)))
print('True Negative = {}'.format(int(a_TN)))

logging.critical('\nAccuracy = {0:1.2f}'.format(a_acc))
print('\nAccuracy = {0:1.2f}'.format(a_acc))
logging.critical('Precision = {0:1.2f}'.format(a_prec))
print('Precision = {0:1.2f}'.format(a_prec))
logging.critical('Recall = {0:1.2f}'.format(a_rec))
print('Recall = {0:1.2f}'.format(a_rec))
logging.critical('F1 Score =  {:1.2f}'.format(a_f1))
print('F1 Score =  {:1.2f}'.format(a_f1))

logging.critical("\nHyper Parameters:")
# print("\nHyper Parameters:")

# Both DB
logging.critical("g_THRESHOLD_UNCERTAINTY {}".format(g_THRESHOLD_UNCERTAINTY))
# print("g_THRESHOLD_UNCERTAINTY {}".format(g_THRESHOLD_UNCERTAINTY))


if chosen_database == db2:
    logging.critical("g_KNN_or_SVC {}".format(g_KNN_or_SVC))
    # print("g_KNN_or_SVC {}".format(g_KNN_or_SVC))
    logging.critical("g_TARGET_TO_DB_NAME_lowerSTD {}".format(g_TARGET_TO_DB_NAME_lowerSTD))
    # print("g_TARGET_TO_DB_NAME_lowerSTD {}".format(g_TARGET_TO_DB_NAME_lowerSTD))
    logging.critical("g_TARGET_TO_DB_NAME_upperSTD {}".format(g_TARGET_TO_DB_NAME_upperSTD))
    # print("g_TARGET_TO_DB_NAME_upperSTD {}".format(g_TARGET_TO_DB_NAME_upperSTD))
    logging.critical("g_UNRECOGNISED_RATIO {}".format(g_UNRECOGNISED_RATIO))
    # print("g_UNRECOGNISED_RATIO {}".format(g_UNRECOGNISED_RATIO))

# Poor DB
if chosen_database == db1:
    logging.critical("g_WEIGHT_avg_dist {}".format(g_WEIGHT_avg_dist))
    # print("g_WEIGHT_avg_dist {}".format(g_WEIGHT_avg_dist))
    logging.critical("g_WEIGHT_min_dist {}".format(g_WEIGHT_min_dist))
    # print("g_WEIGHT_min_dist {}".format(g_WEIGHT_min_dist))
    logging.critical("g_DMv3_lowerSTD {}".format(g_DMv3_lowerSTD))
    # print("g_DMv3_lowerSTD {}".format(g_DMv3_lowerSTD))
    logging.critical("g_DMv3_upperSTD {}".format(g_DMv3_upperSTD))
    # print("g_DMv3_upperSTD {}".format(g_DMv3_upperSTD))
    logging.critical("g_DEFAULT_THRESHOLD {}".format(g_DEFAULT_THRESHOLD))
    # print("g_DEFAULT_THRESHOLD {}".format(g_DEFAULT_THRESHOLD))

time_stop1 = time.time()
time_str = "Session executed in {:1.2f} seconds.".format(time_stop1 - time_start1)
time2_str = "{:1.2f} seconds per one interation.".format((time2_stop - time2_start) / RUNS)
logging.critical(time_str)
print(time_str)
logging.critical(time2_str)
print(time2_str)
