#!/usr/bin/env python3
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.manifold import TSNE



import time
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt  # Plotting tool
import os
from face_recognition.face_verification import FaceVerification
from face_recognition.directory_utils import *

path = os.path.dirname(os.path.abspath(__file__))
FV = FaceVerification()

database_1 = path + '/face_recognition/images/manual_database'  # Option 1
database_2 = path + '/face_recognition/images/mysql_database'   # Option 2

FV.initDatabase(database_2)


dir_path_1 = path + '/face_recognition/images/new_entries/jevgenijs_galaktionovs'
dir_path_2 = path + '/face_recognition/images/new_entries/jesper_bro'
dir_path_3 = path + '/face_recognition/images/new_entries/lelde_skrode'
dir_path_4 = path + '/face_recognition/images/new_entries/hugo_markoff'
dir_path_5 = path + '/face_recognition/images/new_entries/arnold'

db_metadata = FV.database_metadata
db_features = FV.database_features
en_metadata = load_metadata_short(dir_path_1)
en_features, en_metadata = FV.get_features_metadata(en_metadata)

all_metadata = np.append(db_metadata, en_metadata)
all_features = np.append(db_features, en_features)


def distance(feature1, feature2):
    #  Sum of squared errors
    return np.sum(np.square(feature1 - feature2))

def find_threshold(metadata, embedded):
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

    # Plot F1 score and accuracy as function of distance threshold
    plt.plot(thresholds, f1_scores, label='F1 score')
    plt.plot(thresholds, acc_scores, label='Accuracy')
    plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
    plt.title('Accuracy at threshold {0:1.2f} = {1:1.3f}'.format(opt_tau, opt_acc))
    plt.xlabel('Distance threshold')
    plt.legend()
    plt.show()

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

    print('KNN accuracy = {0}, SVM accuracy = {1}'.format(acc_knn, acc_svc))
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

def DoPredict_example(metadata, feature):
    example_image = load_image(metadata.image_path())
    example_prediction = svc.predict(feature)
    example_identity = encoder.inverse_transform(example_prediction)[0]
    print(example_identity)
    plt.imshow(example_image)
    plt.title('Recognized as {0}'.format(example_identity))
    plt.show()

find_threshold(db_metadata, db_features)
test_idx, svc, knn, encoder = init_classifier(db_metadata, db_features)

en_features_reshaped = [item.reshape(1, -1) for item in en_features]

DoPredict_example(en_metadata[5], en_features_reshaped[5])

plot_classifying_results(db_metadata, db_features)