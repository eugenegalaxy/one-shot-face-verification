#!/usr/bin/env python3
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.manifold import TSNE

import operator
import collections
import time
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt  # Plotting tool
import os
from face_recognition.face_verification import FaceVerification
from face_recognition.directory_utils import *


def distance(feature1, feature2):
    #  Sum of squared errors
    return np.sum(np.square(feature1 - feature2))

def find_threshold(metadata, embedded, plot=None):
    distances = []  # squared L2 distance between pairs
    identical = []  # 1 if same identity, 0 otherwise

    num = len(metadata)

    for i in range(num - 1):
        for j in range(i + 1, num):
            distances.append(distance(embedded[i], embedded[j]))
            identical.append(1 if metadata[i].name == metadata[j].name else 0)
            
    distances = np.array(distances)
    identical = np.array(identical)

    thresholds = np.arange(0.3, 1.0, 0.01)

    f1_scores = [f1_score(identical, distances < t) for t in thresholds]
    acc_scores = [accuracy_score(identical, distances < t) for t in thresholds]

    opt_idx = np.argmax(f1_scores)
    # Threshold at maximal F1 score
    opt_tau = thresholds[opt_idx]
    # Accuracy at maximal F1 score
    opt_acc = accuracy_score(identical, distances < opt_tau)
    print('Accuracy at threshold {0:1.2f} = {1:1.3f}'.format(opt_tau, opt_acc))

    if plot:
        # Plot F1 score and accuracy as function of distance threshold
        plt.plot(thresholds, f1_scores, label='F1 score')
        plt.plot(thresholds, acc_scores, label='Accuracy')
        plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
        plt.title('Accuracy at threshold {0:1.2f} = {1:1.3f}'.format(opt_tau, opt_acc))
        plt.xlabel('Distance threshold')
        plt.legend()
        plt.show()

        dist_pos = distances[identical == 1]
        dist_neg = distances[identical == 0]

        plt.figure(figsize=(12,4))
        plt.subplot(121)
        plt.hist(dist_pos)
        plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
        plt.title('Distances (pos. pairs)')
        plt.legend()

        plt.subplot(122)
        plt.hist(dist_neg)
        plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
        plt.title('Distances (neg. pairs)')
        plt.legend()
        plt.show()

    return opt_tau, opt_acc

def init_classifier(metadata, embedded):
    targets = np.array([m.name for m in metadata])

    encoder = LabelEncoder()
    encoder.fit(targets)

    # Numerical encoding of identities
    y = encoder.transform(targets)

    train_idx = np.arange(metadata.shape[0]) % 2 != 0
    test_idx = np.arange(metadata.shape[0]) % 2 == 0

    # 50 train examples of 10 identities (5 examples each)
    X_train = embedded[train_idx]
    # 50 test examples of 10 identities (5 examples each)
    X_test = embedded[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    svc = LinearSVC()

    knn.fit(X_train, y_train)
    svc.fit(X_train, y_train)

    acc_knn = accuracy_score(y_test, knn.predict(X_test))
    acc_svc = accuracy_score(y_test, svc.predict(X_test))

    print('KNN accuracy = {0:1.4f}, SVM accuracy = {1:1.4f}'.format(acc_knn, acc_svc))
    return test_idx, svc, knn, encoder

def plot_classifying_results(metadata, features):
    targets = np.array([m.name for m in metadata])
    X_embedded = TSNE(n_components=2).fit_transform(features)

    for i, t in enumerate(set(targets)):
        idx = targets == t
        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)

    plt.legend(bbox_to_anchor=(1, 1))
    plt.show()


def load_image(path):
    img = cv2.imread(path, 1)
    # return img[..., ::-1]  # Reversing from BGR to RGB
    return img

def avg_dist_target_to_identity(target_metadata, target_feature, database_person_name):
    identify_database_paths = []
    identify_database_features = []

    for idx, item in enumerate(db_metadata):
        if item.name == database_person_name:
            identify_database_paths.append(item)
            identify_database_features.append(db_features[idx])

    all_dists = []
    for idx, item in enumerate(identify_database_paths):
        dist = distance(target_feature, identify_database_features[idx])
        all_dists.append(dist)
        # print('File {0} has {1:1.4f} distance with {2}'.format(
        #   target_metadata.file, dist, os.path.join(item.name, item.file)))

    if len(all_dists) > 1:
        # List without valeus that are small than -2 * std and larger than 1 * std
        lower_std = 1
        upper_std = 1
        all_dists_trimmed = trim_list_std(all_dists, lower_std, upper_std)
        avg_dist = sum(all_dists_trimmed) / len(all_dists_trimmed)
    else:
        avg_dist = sum(all_dists) / len(all_dists)

    return avg_dist


def trim_list_std(list_input, lower_std, upper_std):
    '''
        Note: lower_std -> smaller number trims more lower bracket outliers. 
              upper_std -> smaller number trims more upper bracket outliers.
    '''
    list_input = np.array(list_input)
    mean = np.mean(list_input)
    sd = np.std(list_input)
    final_list = [x for x in list_input if (x > mean - lower_std * sd)]
    final_list = [x for x in final_list if (x < mean + upper_std * sd)]
    return final_list

def predict_once(metadata, feature, threshold, plot=None):
    image = load_image(metadata.image_path())
    prediction = knn.predict(feature)  # returns metadata ID
    identity = encoder.inverse_transform(prediction)[0]

    avg_dist = avg_dist_target_to_identity(metadata, feature, identity)
    THRESHOLD_UNCERTAINTY = 0.15  # TODO/HACK
    if avg_dist < threshold + THRESHOLD_UNCERTAINTY:
        # print('Target {0} recognised {1}'.format(metadata.file, identity))
        if plot:
            plt.imshow(image)
            plt.title('Recognized as {0}'.format(identity))
            plt.show()
        return identity, avg_dist
    else:
        # print('Target unrecognized')
        return 'Unrecognised', 0


path = os.path.dirname(os.path.abspath(__file__))
FV = FaceVerification()

database_1 = path + '/face_recognition/images/manual_database'  # Option 1
database_2 = path + '/face_recognition/images/mysql_database'   # Option 2

db_path = database_2
FV.initDatabase(db_path)
db_metadata = FV.database_metadata
db_features = FV.database_features

dir_path_1 = path + '/face_recognition/images/new_entries/jevgenijs_galaktionovs'
dir_path_2 = path + '/face_recognition/images/new_entries/jesper_bro'
dir_path_3 = path + '/face_recognition/images/new_entries/lelde_skrode'
dir_path_4 = path + '/face_recognition/images/new_entries/hugo_markoff'
dir_path_5 = path + '/face_recognition/images/new_entries/Vladimir_Putin'

tg_path = dir_path_5
en_metadata = load_metadata_short(tg_path)
en_features, en_metadata = FV.get_features_metadata(en_metadata)

all_metadata = np.append(db_metadata, en_metadata)
all_features = np.append(db_features, en_features)
opt_thr, opt_acc = find_threshold(db_metadata, db_features)
test_idx, svc, knn, encoder = init_classifier(db_metadata, db_features)
en_features_reshaped = [item.reshape(1, -1) for item in en_features]

# plot_classifying_results(db_metadata, db_features)

identity_list = []
avg_dist_list = []
for x in range(len(en_features_reshaped)):
    identity, avg_dist = predict_once(en_metadata[x], en_features_reshaped[x], opt_thr)
    identity_list.append(identity)
    avg_dist_list.append(avg_dist)

identity_list, avg_dist_list = zip(*sorted(zip(identity_list, avg_dist_list)))  # Sort two lists in sync
identity_list, avg_dist_list = (list(t) for t in zip(*sorted(zip(identity_list, avg_dist_list))))

# ====== Getting average distance for each identity that was recognised
result_dict = {}
for idx, item in enumerate(identity_list):
    if item not in result_dict.keys():
        result_dict[item] = []

    if item in result_dict.keys():
        result_dict[item].append(avg_dist_list[idx])

for item in result_dict:
    result_dict[item] = [len(result_dict[item]), (sum(result_dict[item]) / len(result_dict[item]))]
# =========

# CASE 1: If nothing has been recognized (only 'Unrecognized' key in dict)
if len(result_dict) == 1 and 'Unrecognised' in result_dict:
    target_info = {
        'fullName': 'Unrecognized',
        'languageCode': 'en',
        'voiceRec': 'en-US'
    }
    final_name = 'Unrecognized'

else:
    #  CASE 2: If dict contains identities AND 'Unrecognized' -> process with identities.
    if 'Unrecognised' in result_dict:
        result_dict.pop('Unrecognised')

    largest_count = 0
    for item in result_dict:
        #  CASE 3: If dict contains more than 1 recognized target -> pick one with more photos recognised.
        if result_dict[item][0] > largest_count:
            largest_count = result_dict[item][0]
            final_name = item
        #  CASE 4: If dict contains more than 1 recognized target with SAME number of recognised photos -> pick smallest avg dist
        elif result_dict[item][0] == largest_count:
            print('whoops, {0} and {1} have the same number of recognised photos. Take smallest avg distance.'.format(
                item, final_name))
            min_avg_dist = min(result_dict[item][1], result_dict[final_name][1])
            if min_avg_dist == result_dict[item][1]:
                largest_count = result_dict[item][0]
                final_name = item
            else:
                largest_count = result_dict[final_name][0]

print(final_name)
