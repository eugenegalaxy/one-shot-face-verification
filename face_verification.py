import cv2
import numpy as np
import matplotlib.pyplot as plt  # Plotting tool
from align import AlignDlib  # Face alignment method
from model import create_model  # CNN library
from directory_utils import load_metadata, find_language_code
from camera_module import getWebcamImage


RS_CAM_AVAILABLE = False

if RS_CAM_AVAILABLE is True:
    from camera_module import rs_init_camera


class FaceVerification(object):

    landmarks_pt = 'models/landmarks.dat'
    weights_pt = 'weights/nn4.small2.v1.h5'
    newtarget_pt = 'new_entries'
    threshold = 0.75

    def __init__(self):
        self.nn4_small2_pretrained = create_model()  # Create a Neural network model
        self.nn4_small2_pretrained.load_weights(self.weights_pt)  # Use pre-trained weights
        self.alig_model = AlignDlib(self.landmarks_pt)  # Initialize the OpenFace face alignment utility

        if RS_CAM_AVAILABLE is True:
            self.pipeline = rs_init_camera()

    if RS_CAM_AVAILABLE is True:
        def __del__(self):
            if self.pipeline in locals():
                self.pipeline.stop()

    def doPrediction(self, plot=None):

        # fresh_image = getWebcamImage(save=1)
        fresh_image = self.load_image('new_entries/image_0003.jpg')
        self.target_features = self.get_features_img(fresh_image)

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
        self.database_features = self.get_features_metadata(self.database_metadata)

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
        for i, m in enumerate(metadata):
            img = self.load_image(m.image_path())
            aligned_img = self.align_image(img)
            if aligned_img is None:
                embedded[i] = np.full((1, 128), -0.1)  # HACK. Giving bad value so distance will never be small.
                print('Cannot locate face on {} image. Skipping.'.format(m.image_path()))
            else:
                aligned_img = (aligned_img / 255.).astype(np.float32)  # scale RGB values to interval [0,1]
                embedded[i] = self.nn4_small2_pretrained.predict(np.expand_dims(aligned_img, axis=0))[0]
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


FV = FaceVerification()
database_pt = 'target_database'
FV.init_database(database_pt, target_names=1)
FV.doPrediction(plot=1)
