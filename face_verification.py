import time
import cv2
import numpy as np
import matplotlib.pyplot as plt  # Plotting tool
from align import AlignDlib  # Face alignment method
from model import create_model  # CNN library
from directory_utils import load_metadata, load_metadata_short, find_language_code

# If Intel Real sense camera is connected, set to True. Set False for Webcamera.
RS_CAM_AVAILABLE = False

# ------ Image Modes -------
GET_FRESH_CAM_IMAGE = 0
ALL_FROM_DIRECTORY = 1
SINGLE_IMAGE_PATH = 2

# ------ Directories -------
database_pt = 'target_database'
new_entries_pt = 'new_entries'
scanned_entries_pt = 'scanned_entries'

if RS_CAM_AVAILABLE is True:
    from camera_module import rs_getImage as getImage
else:
    from camera_module import getWebcamImage as getImage


class FaceVerification(object):

    landmarks_pt = 'models/landmarks.dat'
    weights_pt = 'weights/nn4.small2.v1.h5'
    newtarget_pt = 'new_entries'
    threshold = 0.75

    def __init__(self, img_mode=GET_FRESH_CAM_IMAGE):
        self.nn4_small2_pretrained = create_model()  # Create a Neural network model
        self.nn4_small2_pretrained.load_weights(self.weights_pt)  # Use pre-trained weights
        self.alig_model = AlignDlib(self.landmarks_pt)  # Initialize the OpenFace face alignment utility
        self.img_mode = img_mode  # Image acquisition mode: all images from folder = 1, fresh capture = 2, single image from path = 3

    if RS_CAM_AVAILABLE is True:
        def __del__(self):
            if self.pipeline in locals():
                self.pipeline.stop()

    def doPrediction(self, img_path=None, plot=None):

        if self.img_mode == GET_FRESH_CAM_IMAGE:
            fresh_image = getImage(save=1)

            target_features = self.get_features_img(fresh_image)
            all_dist, min_dist, min_idx = self.dist_target_to_database(target_features)
            listed_score = list(enumerate(all_dist, start=1))
            for item in listed_score:
                print('Target score: {}'.format(item))

        elif self.img_mode == SINGLE_IMAGE_PATH:
            if img_path is None:
                raise ValueError('Parameter img_path is not provided in doPrediction() (Selected mode: SINGLE_IMAGE_PATH')
            fresh_image = cv2.imread(img_path, 1)  # Example path: 'new_entries/image_0000.jpg'
            if fresh_image is None:
                print("============================================\n'{}' does not exist or path is wrong.".format(img_path))
                quit()

            target_features = self.get_features_img(fresh_image)
            all_dist, min_dist, min_idx = self.dist_target_to_database(target_features)
            listed_score = list(enumerate(all_dist, start=1))
            for item in listed_score:
                print('Target score: {}'.format(item))

        elif self.img_mode == ALL_FROM_DIRECTORY:
            pass
        else:
            raise ValueError('Provided img_mode is wrong. Select 0, 1 or 2 and try again.')

        target_features = self.get_features_img(fresh_image)
        all_dist, min_dist, min_idx = self.dist_target_to_database(target_features)
        listed_score = list(enumerate(all_dist, start=1))
        for item in listed_score:
            print('Target score: {}'.format(item))

        target_recognised = self.threshold_check(min_dist)

        if target_recognised is True:
            target_name = self.database_metadata[min_idx].name
            lang_code, lang_str = find_language_code(self.database_metadata[min_idx], print_text=1)
            print("Target recognized as " + str(target_name) + ". Language: " + str(lang_str[0]))

            if plot is not None:
                plt.figure(num="Face Verification", figsize=(8, 5))
                plt.suptitle("Most similar to {0} with Distance of {1:1.3f}\n".format(target_name, min_dist)
                             + "Language code '{0}': {1}.".format(lang_code, lang_str[0]))
                plt.subplot(121)
                fresh_image = fresh_image[..., ::-1]
                plt.imshow(fresh_image)
                plt.subplot(122)
                img2 = self.load_image(self.database_metadata[min_idx].image_path())
                img2 = img2[..., ::-1]
                plt.imshow(img2)
                plt.show()
        else:
            print("Unrecognized person detected!\nSaving image to 'log' folder")

    def init_database(self, path, target_names=None):
        self.database_metadata = load_metadata(path, names=target_names)
        self.database_features, self.database_metadata = self.get_features_metadata(self.database_metadata)

    def get_features_img(self, img):
        embedded = np.zeros(128)
        aligned_img = self.align_image(img)
        if aligned_img is None:
            raise ValueError('Cannot locate face on the image')
        aligned_img = (aligned_img / 255.).astype(np.float32)  # scale RGB values to interval [0,1]
        embedded = self.nn4_small2_pretrained.predict(np.expand_dims(aligned_img, axis=0))[0]
        return embedded

    def get_features_metadata(self, metadata):
        embedded = np.zeros((metadata.shape[0], 128))
        embedded_flt = [0] * len(metadata)
        for i, m in enumerate(metadata):
            img = self.load_image(m.image_path())
            aligned_img = self.align_image(img)
            if aligned_img is None:
                embedded_flt[i] = 1
                print('Cannot locate face in {} -> Excluded from verification'.format(m.image_path()))
            else:
                aligned_img = (aligned_img / 255.).astype(np.float32)  # scale RGB values to interval [0,1]
                embedded[i] = self.nn4_small2_pretrained.predict(np.expand_dims(aligned_img, axis=0))[0]

        bad_idx = [i for i, e in enumerate(embedded_flt) if e == 1]  # indices of photos that failed face-aligning.
        embedded = np.delete(embedded, bad_idx, 0)
        metadata = np.delete(metadata, bad_idx, 0)

        if embedded.shape[0] == 0:
            raise ValueError('Cannot not locate face on any of the images')
        else:
            return embedded, metadata

    def scan_folder(self, path):
        self.entries_metadata = load_metadata_short(path)
        self.entries_features, self.entries_metadata = self.get_features_metadata(self.entries_metadata)
        all_dists = np.zeros((len(self.entries_features), len(self.database_features)))
        min_dists = np.zeros(len(self.entries_features))
        min_index = np.zeros(len(self.entries_features))
        for idx, item in enumerate(self.entries_features):
            all_dists[idx], min_dists[idx], min_index[idx] = self.dist_target_to_database(item)
        dists_avg = np.mean(all_dists, axis=0)
        return dists_avg, min_dists, min_index

        # STOPPED HERE: got avg distance , min distance for each target, index for each target.
        # figure ways to make decision: median of indexes? averages of database? thresholding?
        # get the lowest image
        # do the magic

        # TEST IF single image works with short_metadata

    def dist_target_to_database(self, features):
        distances = []  # squared L2 distance between pairs
        for i in range(len(self.database_features)):
            distances.append(self.distance(self.database_features[i], features))
        distances = np.array(distances)

        min_dist = min(i for i in distances if i > 0)
        # min_dist = np.amin(distances)

        tmp_min_idx = np.where(distances == min_dist)
        min_idx = tmp_min_idx[0][0]

        return distances, min_dist, min_idx

    def load_image(self, path):
        img = cv2.imread(path, 1)
        # return img[..., ::-1]  # Reversing from BGR to RGB
        return img

    def align_image(self, img):
        imgDim = 96
        face_bounding_box = self.alig_model.getLargestFaceBoundingBox(img)
        if face_bounding_box is None:
            return None
        else:
            face_keypoints = AlignDlib.OUTER_EYES_AND_NOSE
            aligned_img = self.alig_model.align(imgDim, img, face_bounding_box, landmarkIndices=face_keypoints)
            return aligned_img

    def distance(self, feature1, feature2):
        #  Sum of squared errors
        return np.sum(np.square(feature1 - feature2))

    def get_language_code(self, metadata, min_idx):
        lang_code, lang_full = find_language_code(metadata[min_idx], print_text=1)
        return lang_code, lang_full

    def threshold_check(self, min_dist):
        if min_dist > self.threshold:
            return False
        else:
            return True


# for x in range(5):
#     getImage(save=1)
#     time.sleep(1)

FV = FaceVerification(img_mode=ALL_FROM_DIRECTORY)
mysql_database_pt = 'mysql_database'
FV.init_database(mysql_database_pt, target_names=1)
a, b, c = FV.scan_folder(new_entries_pt)
print(a)
print(b)
print(c)
# FV.doPrediction(img_path='new_entries/image_0009.jpg',plot=1)
