import cv2
import numpy as np

import matplotlib.pyplot as plt  # Plotting tool
from sklearn.metrics import f1_score, accuracy_score  # Evaluation

# Libraries to classify identity
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.manifold import TSNE  # To plot the results of classifying

from align import AlignDlib  # Face alignment method
from model import create_model  # CNN library
from load_dataset import load_metadata  # Loading training set in from folder

from get_webcam_image import get_webcam_image


def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[..., ::-1]


def align_image(img, alig_model):
    return alig_model.align(96, img, alig_model.getLargestFaceBoundingBox(img),
                            landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)


def run_CNN(metadata, face_align_model):
    embedded = np.zeros((metadata.shape[0], 128))
    for i, m in enumerate(metadata):
        img = load_image(m.image_path())
        img = align_image(img, face_align_model)

        # plt.imshow(img)
        # plt.title('Aligned')
        # plt.show()

        # scale RGB values to interval [0,1]
        img = (img / 255.).astype(np.float32)
        # obtain embedding vector for image
        embedded[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
    return embedded


def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))


def plot_pair_distance_example(metadata, embedded, idx1, idx2):

    dist = distance(embedded[idx1], embedded[idx2])

    plt.figure(figsize=(8, 3))
    plt.suptitle("Distance = {0:1.3f}".format(dist))
    plt.subplot(121)
    plt.imshow(load_image(metadata[idx1].image_path()))
    plt.subplot(122)
    plt.imshow(load_image(metadata[idx2].image_path()))
    plt.show()


def evaluatate_training_data(metadata, embedded):
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

    return opt_tau


def plot_histograms(metadata, embedded, threshold):
    distances = []  # squared L2 distance between pairs
    identical = []  # 1 if same identity, 0 otherwise

    num = len(metadata)

    for i in range(num - 1):
        for j in range(i + 1, num):
            distances.append(distance(embedded[i], embedded[j]))
            identical.append(1 if metadata[i].name == metadata[j].name else 0)

    distances = np.array(distances)
    identical = np.array(identical)

    dist_pos = distances[identical == 1]
    dist_neg = distances[identical == 0]

    plt.figure(figsize=(12, 4))

    plt.subplot(121)
    plt.hist(dist_pos)
    plt.axvline(x=threshold, linestyle='--', lw=1, c='lightgrey', label='Threshold')
    plt.title('Distances (pos. pairs)')
    plt.legend()

    plt.subplot(122)
    plt.hist(dist_neg)
    plt.axvline(x=threshold, linestyle='--', lw=1, c='lightgrey', label='Threshold')
    plt.title('Distances (neg. pairs)')
    plt.legend()
    plt.show()


def init_classifier(metadata, embedded):
    targets = np.array([m.name for m in metadata])

    encoder = LabelEncoder()
    encoder.fit(targets)

    # Numerical encoding of identities
    y = encoder.transform(targets)

    train_idx = np.arange(metadata.shape[0]) % 2 != 0
    print("train_idx: ", train_idx)
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


def DoPredict_example(idx):
    example_image = load_image(database[test_idx][idx].image_path())
    example_prediction = svc.predict([emb_database[test_idx][idx]])
    example_identity = encoder.inverse_transform(example_prediction)[0]

    plt.imshow(example_image)
    plt.title('Recognized as {0}'.format(example_identity))
    plt.show()


def plot_classifying_results(metadata):
    targets = np.array([m.name for m in metadata])
    X_embedded = TSNE(n_components=2).fit_transform(emb_database)

    for i, t in enumerate(set(targets)):
        idx = targets == t
        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)

    plt.legend(bbox_to_anchor=(1, 1))
    plt.show()


def dist_target_to_database(database, emb_database, target, emb_target, plot=None):
    distances = []  # squared L2 distance between pairs
    num = len(database)
    for i in range(num):
        distances.append(distance(emb_database[i], emb_target[0]))
    distances = np.array(distances)

    min_dist = np.amin(distances)
    tmp_min_idx = np.where(distances == np.amin(distances))
    min_idx = tmp_min_idx[0][0]
    most_similar_name = database[min_idx].name

    if plot is not None:
        plt.figure(figsize=(8, 3))
        plt.suptitle("Most similar to {0} with Distance of {1:1.3f}".format(most_similar_name, min_dist))
        plt.subplot(121)
        plt.imshow(load_image(target[0].image_path()))
        plt.subplot(122)
        plt.imshow(load_image(database[min_idx].image_path()))
        plt.show()
    return distances, min_dist, min_idx


# Create a Neural network model (structure from model.py)
nn4_small2_pretrained = create_model()
# Use pre-trained weights
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')
# Initialize the OpenFace face alignment utility
alignment_model = AlignDlib('models/landmarks.dat')

# get_webcam_image()
database = load_metadata('test_images', names=1)
emb_database = run_CNN(database, alignment_model)


# plot_pair_distance_example(database, emb_database, 1, 5)
# opt_threshold = evaluatate_training_data(database, emb_database)
# plot_histograms(database, emb_database, opt_threshold)
# test_idx, svc, knn, encoder = init_classifier(database, emb_database)

path_new_target = 'new_entries'
new_target = load_metadata(path_new_target, names=1)
emb_new_target = run_CNN(new_target, alignment_model)
all_dist, min_dist, min_idx = dist_target_to_database(database, emb_database, new_target, emb_new_target)
print(all_dist)

fake_thr = 0.55

if min_dist > fake_thr:
    print("Unrecognized person detected!")
else:
    most_similar_name = database[min_idx].name
    print("Target recognized as " + str(most_similar_name))
    plt.figure(figsize=(8, 3))
    plt.suptitle("Most similar to {0} with Distance of {1:1.3f}".format(most_similar_name, min_dist))
    plt.subplot(121)
    plt.imshow(load_image(new_target[0].image_path()))
    plt.subplot(122)
    plt.imshow(load_image(database[min_idx].image_path()))
    plt.show()

