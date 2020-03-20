import cv2
import numpy as np
import matplotlib.pyplot as plt  # Plotting tool
from align import AlignDlib  # Face alignment method
from model import create_model  # CNN library
from load_dataset import load_metadata, find_language_code  # Loading training set in from folder
from camera_module import *


class FaceVerification(object):

    landmarks_pt = 'models/landmarks.dat'
    weights_pt = 'weights/nn4.small2.v1.h5'
    database_pt = 'target_database'
    newtarget_pt = 'new_entries'
    threshold = 0.75

    def __init__(self):
        self.nn4_small2_pretrained = create_model()  # Create a Neural network model
        self.nn4_small2_pretrained.load_weights(self.weights_pt)  # Use pre-trained weights
        self.alig_model = AlignDlib(self.landmarks_pt)  # Initialize the OpenFace face alignment utility
        self.init_database(target_names=1)
    #     self.pipeline = rs_init_camera()

    # def __del__(self):
    #     if self.pipeline in locals():
    #         self.pipeline.stop()

    # ---------------------------------- PUBLIC METHODS ----------------------------------
    def doPrediction(self, plot=None):
        # self.target_metadata = load_metadata(self.newtarget_pt)
        # self.target_features = self.extract_features(self.target_metadata)

        fresh_image = get_webcam_image(save=1)
        # fresh_image = load_image('new_entries/person/image.jpg')
        self.target_features = self.test_extract_feature(fresh_image)

        all_dist, min_dist, min_idx = self.dist_target_to_database()
        print(all_dist)
        target_recognised = self.threshold_check(min_dist)

        if target_recognised is True:
            target_name = self.database_metadata[min_idx].name
            lang_code, lang_str = find_language_code(self.database_metadata[min_idx], print_text=1)
            print("Target recognized as " + str(target_name) + ". Language: " + str(lang_str))

            if plot is not None:
                plt.figure(num="Face Verification", figsize=(8, 5))
                plt.suptitle("Most similar to {0} with Distance of {1:1.3f}\n".format(target_name, min_dist)
                             + "Language code '{0}': {1}.".format(lang_code, lang_str))
                plt.subplot(121)
                plt.imshow(fresh_image)
                plt.subplot(122)
                plt.imshow(self.load_image(self.database_metadata[min_idx].image_path()))
                plt.show()
        else:
            print("Unrecognized person detected!\nSaving image to 'log' folder")

    #  ---------------------------------- PRIVATE METHODS ----------------------------------

    def init_database(self, target_names=None):
        self.database_metadata = load_metadata(self.database_pt, names=target_names)
        self.database_features = self.extract_features(self.database_metadata)

    def test_extract_feature(self, img):
        embedded = np.zeros(128)
        time.sleep(1)
        img = self.align_image(img)
        img = (img / 255.).astype(np.float32)  # scale RGB values to interval [0,1]
        embedded = self.nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
        return embedded

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
            distances.append(self.distance(self.database_features[i], self.target_features))
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

# img1 = get_webcam_image(save=1)  # perevernutaja
# print('img1 Dimensions :', img1.shape)
# img1_emb = FV.test_extract_feature(img1)
# time.sleep(5)

# img2 = cv2.imread('new_entries/person/image.jpg', 1)  # ne perevernutaja
# print('img2 Dimensions :', img2.shape)
# img2_emb = FV.test_extract_feature(img2)

# img3 = img2[..., ::-1]  # perevernutaja
# print('img3 Dimensions :', img3.shape)
# img3_emb = FV.test_extract_feature(img3)

# test_img = cv2.imread('target_database/Jevgenijs_Galaktionovs-ru/ae.jpg', 1)
# print('test_img Dimensions :', test_img.shape)
# test_img_emb = FV.test_extract_feature(test_img)

# dist_img1_to_img2 = FV.distance(img1_emb, img2_emb)
# dist_img1_to_img3 = FV.distance(img1_emb, img3_emb)
# dist_img2_to_img3 = FV.distance(img2_emb, img3_emb)
# dist_img3_to_img3 = FV.distance(img3_emb, img3_emb)

# print('Distance img1 to img2: {0:1.4f}'.format(dist_img1_to_img2))
# print('Distance img1 to img3: {0:1.4f}'.format(dist_img1_to_img3))
# print('Distance img2 to img3: {0:1.4f}'.format(dist_img2_to_img3))
# print('Distance img3 to img3: {0:1.4f}'.format(dist_img3_to_img3))
# print('==========================================================')

# dist_img1_to_test = FV.distance(test_img_emb, img1_emb)
# dist_img2_to_test = FV.distance(test_img_emb, img2_emb)
# dist_img3_to_test = FV.distance(test_img_emb, img3_emb)
# print('Distance test to img1: {0:1.4f}'.format(dist_img1_to_test))
# print('Distance test to img2: {0:1.4f}'.format(dist_img2_to_test))
# print('Distance test to img3: {0:1.4f}'.format(dist_img3_to_test))

# plt.figure(num="TEST FV", figsize=(8, 5))
# plt.subplot(121)
# plt.imshow(img1)
# plt.subplot(122)
# plt.imshow(img3)
# plt.show()
