import cv2
import numpy as np
import matplotlib.pyplot as plt  # Plotting tool
import logging

from cnn_module.face_detect import AlignDlib  # Face alignment method
from cnn_module.model import create_model  # CNN library

from directory_utils import load_metadata, load_metadata_short, find_language_code, retrieve_info

# Debug mode. Enables prints.
g_DEBUG_MODE = False

# If Intel Real sense camera is connected, set to True. Set False for Webcamera.
RS_CAM_AVAILABLE = False

if RS_CAM_AVAILABLE is True:
    from camera_module import getImg_realsense as getImg
    from camera_module import getManyImg_realsense as getManyImg

else:
    from camera_module import getImg_webcam as getImg
    from camera_module import getManyImg_webcam as getManyImg

class FaceVerification(object):

    logging.basicConfig(filename='images/verification_info.log',
                        level=logging.DEBUG,
                        # filemode='w',
                        format='%(asctime)s :: %(message)s')
    weights_pt = 'cnn_module/weights/nn4.small2.v1.h5'
    landmarks_pt = 'cnn_module/models/landmarks.dat'

    RECOGNITION_THRESHOLD = 0.65
    img_mode = None  # Image Mode variable

    def __init__(self):
        self.nn4_small2_pretrained = create_model()  # Create a Neural network model
        self.nn4_small2_pretrained.load_weights(self.weights_pt)  # Use pre-trained weights
        self.alig_model = AlignDlib(self.landmarks_pt)  # Initialize the OpenFace face alignment utility

    if RS_CAM_AVAILABLE is True:
        def __del__(self):
            if self.pipeline in locals():
                self.pipeline.stop()

    def getImage(self, save_path=None):
        return getImg(save_path=save_path)

    def getManyImages(self, numberImg, time_interval_sec, save_path):
        return getManyImg(numberImg, time_interval_sec, save_path)

    def initDatabase(self, path, target_names=None):
        self.database_metadata = load_metadata(path, names=target_names)
        self.database_features, self.database_metadata = self.get_features_metadata(self.database_metadata)

    def setImgMode(self, img_mode):
        '''
            Set mode how to obtain NEW images to verify against a database.
            Parameter 'img_mode' options:
            0 (GET_FRESH_CAM_IMAGE): Acquire fresh image and verify it against database.
            1 (SINGLE_IMAGE_PATH): Provide path to image saved on disk and verify it against database.
            2 (ALL_FROM_DIRECTORY): Provide path to directory and scan all images in it (NOTE: No subdirectories!)
        '''
        self.img_mode = img_mode

    def predict(self, single_img_path=None, directory_path=None, plot=None):

        assert (self.img_mode is not None), 'No image mode is selected. See FaceVerification.setImgMode() function.'

        if self.img_mode == 0:
            fresh_image = self.getImg(save_path='images/new_entries')

            target_features = self.get_features_img(fresh_image)
            if type(target_features) == int:
                return -1
            all_dist, min_dist, min_idx = self.dist_target_to_database(target_features)

            if g_DEBUG_MODE is True:
                listed_score = list(enumerate(all_dist, start=1))
                for item in listed_score:
                    print('Target score: {}'.format(item))

        elif self.img_mode == 1:

            assert single_img_path is not None, 'Parameter single_img_path is not provided in predict() (Selected mode: SINGLE_IMAGE_PATH)'

            fresh_image = cv2.imread(single_img_path, 1)  # Example path: 'images/new_entries/image_0000.jpg'

            assert fresh_image is not None, '"{}" does not exist or path is wrong in predict() (Selected mode: SINGLE_IMAGE_PATH)'.format(single_img_path)

            target_features = self.get_features_img(fresh_image)
            if type(target_features) == int:
                return -1
            all_dist, min_dist, min_idx = self.dist_target_to_database(target_features)

            if g_DEBUG_MODE is True:
                listed_score = list(enumerate(all_dist, start=1))
                for item in listed_score:
                    print('Target score: {}'.format(item))

        elif self.img_mode == 2:

            assert directory_path is not None, 'Parameter directory_path is not provided in predict() (Selected mode: ALL_FROM_DIRECTORY)'

            self.entries_metadata = load_metadata_short(directory_path)
            self.entries_features, self.entries_metadata = self.get_features_metadata(self.entries_metadata)
            all_dists = np.zeros((len(self.entries_features), len(self.database_features)))
            min_dists = np.zeros(len(self.entries_features))
            min_idxs = np.zeros(len(self.entries_features))
            for idx, item in enumerate(self.entries_features):
                all_dists[idx], min_dists[idx], min_idxs[idx] = self.dist_target_to_database(item)
            avg_dists = np.mean(all_dists, axis=0)

            if g_DEBUG_MODE is True:
                print("================ ALL DISTANCES ================")
                print(all_dists)
                print("============== AVERAGE DISTANCES ==============")
                print(avg_dists)
                print("============== MINIMUM DISTANCES ==============")
                print(min_dists)
                print("========= INDEX OF MINIMUM DISTANCES ==========")
                print(min_idxs)

            min_dist, min_idx, img_idx = self.decision_maker_v2(all_dists, avg_dists, min_dists, min_idxs)
            fresh_image = self.entries_metadata[img_idx]  # NOTE: Crappy name -> for sake of consisency with other img_mode cases above.
            fresh_image = self.load_image(fresh_image.image_path())

        target_recognised = self.threshold_check(min_dist)

        if target_recognised is True:
            target_base = self.database_metadata[min_idx].base
            target_name = self.database_metadata[min_idx].name
            target_path = target_base + '/' + target_name
            target_info, target_images = retrieve_info(target_path)
            lang_code, lang_str = find_language_code(self.database_metadata[min_idx])

            if g_DEBUG_MODE is True:
                print("Target recognized as {0} with {1:1.3f} score. Language: {2}".format(target_info['fullName'], min_dist, target_info['nationality']))

            logging.info("Target recognized as {0}. Language: {1}".format(str(target_name), target_info['nationality']))

            if plot is not None:
                plt.figure(num="Face Verification", figsize=(8, 5))
                title_part_1 = "Most similar to {0} with Distance of {1:1.3f}\n".format(target_name, min_dist)
                title_part_2 = "Language code '{0}': {1}.".format(target_info['languageCode'], target_info['nationality'])
                plt.suptitle(title_part_1 + title_part_2)
                plt.subplot(121)
                fresh_image = fresh_image[..., ::-1]
                plt.imshow(fresh_image)
                plt.subplot(122)
                img2 = self.load_image(self.database_metadata[min_idx].image_path())
                img2 = img2[..., ::-1]
                plt.imshow(img2)
                plt.show()

            return target_info, target_images
        else:
            if g_DEBUG_MODE is True:
                print("Unrecognized person detected.")

            logging.info("Unrecognized person detected.")
            if plot is not None:
                plt.figure(num="Face Verification", figsize=(8, 5))
                plt.suptitle("Who's that Pokemon?\n(Unrecognized person detected with {0:1.3f} distance score)".format(min_dist))
                plt.subplot(121)
                fresh_image = fresh_image[..., ::-1]
                plt.imshow(fresh_image)
                plt.subplot(122)
                img2 = self.load_image('useful_stuff/surprise.jpg')
                img2 = img2[..., ::-1]
                plt.imshow(img2)
                plt.show()

            target_info = {
                'fullName': 'Unrecognized',
                'languageCode': 'en',
                'voiceRec': 'en-US'
            }
            target_images = []
            return target_info, target_images

    def get_features_img(self, img):
        embedded = np.zeros(128)
        aligned_img = self.align_image(img)
        if aligned_img is None:
            # raise ValueError('Cannot locate face on the image')
            print('Cannot locate face on image.')
            return -1
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
                if g_DEBUG_MODE is True:
                    print('Cannot locate face in {} -> Excluded from verification'.format(m.image_path()))
                logging.info('Cannot locate face in {} -> Excluded from verification'.format(m.image_path()))
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

    def decision_maker_v1(self, all_dists, avg_dists, min_dists, min_idxs):
        min_val = np.min(avg_dists)
        min_index = np.where(avg_dists == min_val)
        new_min_list = []
        [new_min_list.append(sublist[min_index[0][0]]) for sublist in all_dists]
        min_index_sublists = new_min_list.index(min(new_min_list))
        lowest_score = all_dists[min_index_sublists][min_index[0][0]]
        return lowest_score, min_index[0][0], min_index_sublists

    def decision_maker_v2(self, all_dists, avg_dists, min_dists, min_idxs):
        lowest_score = 5  # just a big number for distance scores
        for idx1, item in enumerate(all_dists):
            for idx2, score in enumerate(item):
                if score < lowest_score:
                    lowest_score = score
                    lowest_score_idx = idx2
                    lowest_image_idx = idx1

        # min_val = np.min(avg_dists)
        # min_index = np.where(avg_dists == min_val)
        # new_min_list = []
        # [new_min_list.append(sublist[min_index[0][0]]) for sublist in all_dists]
        # min_index_sublists = new_min_list.index(min(new_min_list))
        # lowest_score = all_dists[min_index_sublists][min_index[0][0]]
        return lowest_score, lowest_score_idx, lowest_image_idx

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

    def threshold_check(self, min_dist):
        if min_dist > self.RECOGNITION_THRESHOLD:
            return False
        else:
            return True
