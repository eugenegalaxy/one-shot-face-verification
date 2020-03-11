import cv2
import numpy as np
import time
import matplotlib.pyplot as plt  # Plotting tool

from align import AlignDlib  # Face alignment method
from model import create_model  # CNN library
from load_dataset import load_metadata, find_language_code  # Loading training set in from folder
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


nn4_small2_pretrained = create_model()  # Create a Neural network model
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')  # Use pre-trained weights
alignment_model = AlignDlib('models/landmarks.dat')  # Initialize the OpenFace face alignment utility

database = load_metadata('target_database', names=1)
emb_database = run_CNN(database, alignment_model)

# get_webcam_image()
# time.sleep(5)
path_new_target = 'new_entries'
new_target = load_metadata(path_new_target, names=1)
emb_new_target = run_CNN(new_target, alignment_model)
all_dist, min_dist, min_idx = dist_target_to_database(database, emb_database, new_target, emb_new_target)
print(all_dist)
fake_thr = 0.65
lang_code, lang_full = find_language_code(database[min_idx], print_text=1)

if min_dist > fake_thr:
    print("Unrecognized person detected!")
else:
    most_similar_name = database[min_idx].name
    print("Target recognized as " + str(most_similar_name))

    plt.figure(figsize=(8, 3))
    plt.suptitle("Most similar to {0} with Distance of {1:1.3f}\n \
                  Language code '{2}': {3}.".format(most_similar_name, min_dist, lang_code, lang_full[0]))
    plt.subplot(121)
    plt.imshow(load_image(new_target[0].image_path()))
    plt.subplot(122)
    plt.imshow(load_image(database[min_idx].image_path()))
    plt.show()
