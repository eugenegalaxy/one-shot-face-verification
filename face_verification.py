import cv2
import numpy as np
import matplotlib.pyplot as plt  # Plotting tool

from align import AlignDlib  # Face alignment method
from model import create_model  # CNN library
from load_dataset import load_metadata, find_language_code  # Loading training set in from folder
# from get_webcam_image import get_webcam_image


class FaceVerification(object):

    landmarks_pt = 'models/landmarks.dat'
    weights_pt = 'weights/nn4.small2.v1.h5'
    database_pt = 'target_database'
    newtarget_pt = 'new_entries'
    threshold = 0.65

    def __init__(self):
        self.nn4_small2_pretrained = create_model()  # Create a Neural network model
        self.nn4_small2_pretrained.load_weights(self.weights_pt)  # Use pre-trained weights
        self.alig_model = AlignDlib(self.landmarks_pt)  # Initialize the OpenFace face alignment utility
        self.init_database()

    # ---------------------------------- PUBLIC METHODS ----------------------------------
    def doPrediction(self, plot=None):
        self.target_metadata = load_metadata(self.newtarget_pt)
        self.target_features = self.extract_features(self.target_metadata)

        all_dist, min_dist, min_idx = self.dist_target_to_database()

        target_recognised = self.threshold_check(min_dist)

        if target_recognised is True:
            target_name = self.database_metadata[min_idx].name
            lang_code, lang_str = find_language_code(self.database_metadata[min_idx], print_text=1)
            print("Target recognized as " + str(target_name) + ". Language: " + str(lang_str))

            if plot is not None:
                plt.figure(num="Face Verification", figsize=(8, 4))
                plt.suptitle("Most similar to {0} with Distance of {1:1.3f}\n".format(target_name, min_dist)
                             + "Language code '{0}': {1}.".format(lang_code, lang_str))
                plt.subplot(121)
                plt.imshow(self.load_image(self.target_metadata[0].image_path()))
                plt.subplot(122)
                plt.imshow(self.load_image(self.database_metadata[min_idx].image_path()))
                plt.show()
        else:
            print("Unrecognized person detected!\nSaving image to 'log' folder")

    #  ---------------------------------- PRIVATE METHODS ----------------------------------

    def init_database(self):
        self.database_metadata = load_metadata(self.database_pt, names=1)
        self.database_features = self.extract_features(self.database_metadata)

    def extract_features(self, metadata):
        embedded = np.zeros((metadata.shape[0], 128))
        for i, m in enumerate(metadata):
            img = self.load_image(m.image_path())
            img = self.align_image(img)
            img = (img / 255.).astype(np.float32)  # scale RGB values to interval [0,1]
            embedded[i] = self.nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
        return embedded

    def dist_target_to_database(self):
        distances = []  # squared L2 distance between pairs
        num = len(self.database_metadata)
        for i in range(num):
            distances.append(self.distance(self.database_features[i], self.target_features[0]))
        distances = np.array(distances)

        min_dist = np.amin(distances)
        tmp_min_idx = np.where(distances == np.amin(distances))
        min_idx = tmp_min_idx[0][0]
        return distances, min_dist, min_idx

    def load_image(self, path):
        img = cv2.imread(path, 1)
        return img[..., ::-1]  # Reversing from BGR to RGB

    def align_image(self, img):
        imgDim = 96
        face_bounding_box = self.alig_model.getLargestFaceBoundingBox(img)
        face_keypoints = AlignDlib.OUTER_EYES_AND_NOSE
        aligned_img = self.alig_model.align(imgDim, img, face_bounding_box, landmarkIndices=face_keypoints)
        return aligned_img

    def distance(self, feature1, feature2):
        return np.sum(np.square(feature1 - feature2))

    def get_language_code(self, metadata, min_idx):
        lang_code, lang_full = find_language_code(metadata[min_idx], print_text=1)
        return lang_code, lang_full

    def threshold_check(self, min_dist):
        if min_dist > self.threshold:
            return False
        else:
            return True


FV = FaceVerification()
FV.doPrediction(plot=1)
