#!/usr/bin/env python3
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder


from face_recognition.cnn_module.face_detect import AlignDlib  # Face alignment method
from face_recognition.cnn_module.model import create_model  # CNN library

from face_recognition.directory_utils import load_metadata, load_metadata_short, \
    retrieve_info, IdentityMetadata, IdentityMetadata_short, resize_img, trim_list_std


#  Both DB
g_THRESHOLD_UNCERTAINTY = 0  # Offset to threshold. Decrease if strangers recognized as identities

# Rich DB
g_KNN_or_SVC = 0  # 0 for KNN, 1 for SVC
g_TARGET_TO_DB_NAME_lowerSTD = 2  # Don't go lower than 1
g_TARGET_TO_DB_NAME_upperSTD = 2  # Don't go lower than 1
g_UNRECOGNISED_RATIO = 0.8  # Ratio of unrecognised to others to count target = Unrecognized.

# Poor DB
g_WEIGHT_avg_dist = 0.5
g_WEIGHT_min_dist = 0.5
g_DMv3_lowerSTD = 1
g_DMv3_upperSTD = 3
g_DEFAULT_THRESHOLD = 0.55  # Default recognition threshold.


g_DEBUG_MODE = True  # Debug mode. Enables prints.
g_LOGGER_ENABLE = False
g_RS_CAM_AVAILABLE = False  # If Intel Real sense camera is connected, set to True. Set False for Webcamera.

if g_RS_CAM_AVAILABLE is True:
    from face_recognition.camera_module import getImg_realsense as getImg
    from face_recognition.camera_module import getManyImg_realsense as getManyImg
else:
    from face_recognition.camera_module import getImg_webcam as getImg
    from face_recognition.camera_module import getManyImg_webcam as getManyImg

if g_LOGGER_ENABLE is True:
    import logging


class FaceVerification(object):

    path = os.path.dirname(os.path.abspath(__file__))

    if g_LOGGER_ENABLE is True:
        logging.basicConfig(filename=path + '/images/verification_info.log',
                            level=logging.DEBUG,
                            # filemode='w',
                            format='%(asctime)s :: %(message)s')

    weights_pt = path + '/cnn_module/weights/nn4.small2.v1.h5'
    landmarks_pt = path + '/cnn_module/models/landmarks.dat'

    classifier_valid = False  # Variable that decides if classifier will be used (computed)
    SVC = None  # Classificator variable
    KNN = None
    classifier_encoder = None  # SVC or KNN encoder variable

    img_mode = None  # img Mode variable

    def __init__(self):
        self.nn4_small2_pretrained = create_model()  # Create a Neural network model
        self.nn4_small2_pretrained.load_weights(self.weights_pt)  # Use pre-trained weights
        self.alig_model = AlignDlib(self.landmarks_pt)  # Initialize the OpenFace face alignment utility

    if g_RS_CAM_AVAILABLE is True:
        def __del__(self):
            if self.pipeline in locals():
                self.pipeline.stop()

    def getImg(self, save_path=None):
        return getImg(save_path=save_path)

    def getManyImg(self, numberImg, time_interval_sec, save_path):
        return getManyImg(numberImg, time_interval_sec, save_path)

    def initdb_and_classifier(self, path, tg_names=None):
        self.db_metadata = load_metadata(path, names=tg_names)
        self.db_features, self.db_metadata = self.get_features_metadata(self.db_metadata)
        self.train_classifier(self.db_metadata, self.db_features)
        self.threshold = self.find_threshold(self.db_metadata, self.db_features)

    def setImgMode(self, img_mode):
        '''
            Set mode how to obtain NEW imgs to verify against a database.
            Parameter 'img_mode' options:
            0 (GET_FRESH_CAM_img): Acquire fresh img and verify it against database.
            1 (SINGLE_img_PATH): Provide path to img saved on disk and verify it against database.
            2 (ALL_FROM_DIRECTORY): Provide path to directory and scan all imgs in it (NOTE: No subdirectories!)
        '''
        self.img_mode = img_mode

    def Predict(self, single_img_path=None, directory_path=None, plot=None):

        assert (self.img_mode is not None), 'No img mode is selected. See FaceVerification.setImgMode() function.'

        if self.img_mode == 0 or self.img_mode == 1:

            if self.img_mode == 0:
                path_to_save = 'images/new_entries'
                full_path = os.path.join(self.path, path_to_save)
                fresh_img = self.getImg(save_path=full_path)

            elif self.img_mode == 1:
                assert single_img_path is not None, 'Parameter single_img_path is not provided in predict() \
                                                    (Selected mode: SINGLE_img_PATH)'

                fresh_img = cv2.imread(single_img_path, 1)  # Example path: 'imgs/new_entries/img_0000.jpg'

                assert fresh_img is not None, '"{}" does not exist or path is wrong in predict() \
                                                (Selected mode: SINGLE_img_PATH)'.format(single_img_path)

            tg_features = self.get_features_img(fresh_img)

            if type(tg_features) == int:
                raise TypeError('Cannot detect face on a fresh image...')

            all_dist, min_dist, min_idx = self.dist_tg_to_db(tg_features)

            tg_recognised = self.threshold_check(min_dist)
            if tg_recognised is True:
                db_path = os.path.join(self.db_metadata[min_idx].base, self.db_metadata[min_idx].name)

            if g_DEBUG_MODE is True:
                for numb, dist in enumerate(all_dist):
                    print('File-> Dist: {0:1.3f} to img -> {1}. DB Index: {2} '.format(
                        dist, os.path.join(self.db_metadata[numb].name, self.db_metadata[numb].file), numb))

        elif self.img_mode == 2:

            assert directory_path is not None, 'Parameter directory_path is not provided in predict() \
                                                (Selected mode: ALL_FROM_DIRECTORY)'

            self.tg_metadata = load_metadata_short(directory_path)
            self.tg_features, self.tg_metadata = self.get_features_metadata(self.tg_metadata)

            if self.classifier_valid is True:
                name, min_dist = self.compute_identity(self.tg_metadata, self.tg_features, self.threshold)

                fresh_img = self.load_img(self.tg_metadata[0].img_path())
                if name == 'Unrecognised':
                    tg_recognised = False
                else:
                    tg_recognised = True
                    for item in self.db_metadata:
                        if item.name == name:
                            db_path = os.path.join(item.base, item.name)

            else:
                all_dists = np.zeros((len(self.tg_features), len(self.db_features)))
                min_dists = np.zeros(len(self.tg_features))
                min_idxs = np.zeros(len(self.tg_features))
                for idx, item in enumerate(self.tg_features):
                    all_dists[idx], min_dists[idx], min_idxs[idx] = self.dist_tg_to_db(item)
                thr_min_dist, min_idx, img_idx = self.decision_maker_v3(all_dists)

                min_dist = min_dists[img_idx]  # NOTE Crappy name -> consisency with plotting names
                fresh_img = self.tg_metadata[img_idx]  # NOTE: Crappy name -> consisency with other img_mode cases
                fresh_img = self.load_img(fresh_img.img_path())

                if g_DEBUG_MODE is True:
                    print("============== MINIMUM DISTANCES ==============")
                    for numb, dist in enumerate(min_dists):
                        idx = int(min_idxs[numb])
                        print('File {0}: {1} -> Min Dist: {2:1.3f} to img -> {3}. DB Index: {4} '.format(
                            numb, self.tg_metadata[numb].file, dist,
                            os.path.join(self.db_metadata[idx].name, self.db_metadata[idx].file), idx))

                thr_min_dist -= g_THRESHOLD_UNCERTAINTY  # HACK
                tg_recognised = self.threshold_check(thr_min_dist)
                if tg_recognised is True:
                    db_path = os.path.join(self.db_metadata[min_idx].base, self.db_metadata[min_idx].name)
                    name = self.db_metadata[min_idx].name

        if tg_recognised is True:

            tg_info, db_imgs = retrieve_info(db_path)
            if g_DEBUG_MODE is True:
                print("Target recognised as {0} with {1:1.3f} score. Language: {2} \
                      ".format(tg_info['fullName'], min_dist, tg_info['nationality']))
            if g_LOGGER_ENABLE is True:
                logging.info("Target recognised as {0} with {1:1.3f} score. Language: {2} \
                      ".format(tg_info['fullName'], min_dist, tg_info['nationality']))
            if plot is not None:
                plt.figure(num="Face Verification", figsize=(8, 5))
                title_1 = "Most similar to {0} with Distance of {1:1.3f}\n".format(
                    tg_info['fullName'], min_dist)
                title_2 = "Language code '{0}': {1}.".format(
                    tg_info['languageCode'], tg_info['nationality'])
                plt.suptitle(title_1 + title_2)
                plt.subplot(121)
                fresh_img = fresh_img[..., ::-1]
                plt.imshow(fresh_img)
                plt.subplot(122)
                img2 = self.load_img(db_imgs[0])
                img2 = img2[..., ::-1]
                plt.imshow(img2)
                plt.show()
            return tg_info, db_imgs
        else:
            if g_DEBUG_MODE is True:
                print("Unrecognised person detected.")
            if g_LOGGER_ENABLE is True:
                logging.info("Unrecognised person detected.")
            if plot is not None:
                plt.figure(num="Face Verification", figsize=(8, 5))
                plt.suptitle("Who's that Pokemon?\n(Unrecognised person detected)")
                plt.subplot(121)
                fresh_img = fresh_img[..., ::-1]
                plt.imshow(fresh_img)
                plt.subplot(122)
                img2 = self.load_img('face_recognition/useful_stuff/surprise.jpg')
                img2 = img2[..., ::-1]
                plt.imshow(img2)
                plt.show()

            tg_info = {
                'fullName': 'Unrecognised',
                'languageCode': 'en',
                'voiceRec': 'en-US'
            }
            return tg_info, []

    def get_features_img(self, img):
        embedded = np.zeros(128)
        if img.shape[0] > 500:
            img = resize_img(img, adjust_to_width=500)
        aligned_img = self.align_img(img)
        if aligned_img is None:
            return -1
        aligned_img = (aligned_img / 255.).astype(np.float32)  # scale RGB values to interval [0,1]
        embedded = self.nn4_small2_pretrained.predict(np.expand_dims(aligned_img, axis=0))[0]
        return embedded

    def get_features_metadata(self, metadata):
        embedded = np.zeros((metadata.shape[0], 128))
        embedded_flt = [0] * len(metadata)
        for i, m in enumerate(metadata):

            features = self.read_features_from_disk(m)
            if features is not None:
                embedded[i] = features
            else:
                img = self.load_img(m.img_path())
                if img.shape[1] > 500:
                    img = resize_img(img, adjust_to_width=700)
                aligned_img = self.align_img(img)
                if aligned_img is None:
                    embedded_flt[i] = 1
                    if g_DEBUG_MODE is True:
                        print('Cannot locate face in {} -> Excluded from verification'.format(m.img_path()))
                    if g_LOGGER_ENABLE is True:
                        logging.info('Cannot locate face in {} -> Excluded from verification'.format(m.img_path()))
                else:
                    aligned_img = (aligned_img / 255.).astype(np.float32)  # scale RGB values to interval [0,1]
                    embedded[i] = self.nn4_small2_pretrained.predict(np.expand_dims(aligned_img, axis=0))[0]
                    self.save_features_on_disk(m, embedded[i])
        bad_idx = [i for i, e in enumerate(embedded_flt) if e == 1]  # indices of photos that failed face-aligning.
        embedded = np.delete(embedded, bad_idx, 0)
        metadata = np.delete(metadata, bad_idx, 0)

        if embedded.shape[0] == 0:
            raise ValueError('Cannot not locate face on any of the imgs')
        else:
            return embedded, metadata

    def decision_maker_v3(self, all_dists):
        lower_std = g_DMv3_lowerSTD
        upper_std = g_DMv3_upperSTD
        all_dists_trimmed = []
        all_dists = np.array(all_dists)
        all_dists_tr = np.transpose(all_dists)

        trimmed_list = []
        for idx, sub_list in enumerate(all_dists_tr):
            if len(sub_list) <= 1:
                trimmed_list.append(sub_list)
            else:
                tmp_trim = []
                while len(tmp_trim) == 0:
                    tmp_trim = trim_list_std(sub_list, lower_std, upper_std)
                    lower_std += 0.05
                    upper_std += 0.02
                trimmed_list.append(tmp_trim)

        avg_dists = []
        min_dists = []
        for sub_list in trimmed_list:
            avg_dists.append(sum(sub_list) / len(sub_list))
            min_dists.append(min(sub_list))

        avg_dists_minus_thr_uncrt = [item - g_THRESHOLD_UNCERTAINTY for item in avg_dists]  # HACK

        weighted_combined_list = []
        for x in range(len(avg_dists_minus_thr_uncrt)):
            element_sum = (min_dists[x] * g_WEIGHT_min_dist) + (avg_dists_minus_thr_uncrt[x] * g_WEIGHT_avg_dist)
            weighted_combined_list.append(element_sum)
        if g_DEBUG_MODE is True:
            print('=========== Weighted-Combined scores ===========')
            print(weighted_combined_list)
        lowest_score = min(weighted_combined_list)
        lowest_db_id = weighted_combined_list.index(lowest_score)
        lowest_tg_id_tmp = np.where(all_dists_tr[lowest_db_id] == min(all_dists_tr[lowest_db_id]))
        lowest_tg_id = lowest_tg_id_tmp[0][0]
        return lowest_score, lowest_db_id, lowest_tg_id

    def dist_tg_to_db(self, features):
        distances = []  # squared L2 distance between pairs
        for i in range(len(self.db_features)):
            distances.append(self.distance(self.db_features[i], features))
        distances = np.array(distances)

        min_dist = min(i for i in distances if i > 0)
        # min_dist = np.amin(distances)

        tmp_min_idx = np.where(distances == min_dist)
        min_idx = tmp_min_idx[0][0]

        return distances, min_dist, min_idx

    def load_img(self, path):
        img = cv2.imread(path, 1)
        # return img[..., ::-1]  # Reversing from BGR to RGB
        return img

    def align_img(self, img):
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
        if min_dist > g_DEFAULT_THRESHOLD:
            return False
        else:
            return True

    def save_features_on_disk(self, metadata, features, overwrite_txt=None):
        features_str = str(features)
        if isinstance(metadata, IdentityMetadata):
            parent_path = os.path.join(metadata.base, metadata.name)
        elif isinstance(metadata, IdentityMetadata_short):
            parent_path = metadata.base
        else:
            raise TypeError('Argument "metadata" is neither IdentityMetadata or IdentityMetadata_short type.')

        file_name_no_ext = os.path.splitext(metadata.file)[0]
        file_name_ext_txt = file_name_no_ext + '.txt'
        path_file = os.path.join(parent_path, file_name_ext_txt)

        if file_name_ext_txt in os.listdir(parent_path) and overwrite_txt is None:
            if g_DEBUG_MODE is True:
                print('{} already exists'.format(file_name_ext_txt))
        else:
            with open(path_file, "w") as file:
                file.write(features_str)
                file.close()

    def read_features_from_disk(self, metadata):
        if isinstance(metadata, IdentityMetadata):
            parent_path = os.path.join(metadata.base, metadata.name)
        elif isinstance(metadata, IdentityMetadata_short):
            parent_path = metadata.base
        else:
            raise TypeError('Argument "path" is neither IdentityMetadata or IdentityMetadata_short type.')

        file_name_no_ext = os.path.splitext(metadata.file)[0]
        file_name_ext_txt = file_name_no_ext + '.txt'
        path_file = os.path.join(parent_path, file_name_ext_txt)
        if os.path.exists(path_file):
            with open(path_file) as f:
                text = f.read()
                text = text.split()
                text[0] = text[0].replace('[', '')
                text[-1] = text[-1].replace(']', '')
                for item in text:
                    try:
                        item = float(item)
                    except ValueError:
                        text.remove(item)
                text = np.array(text)
            return text
        else:
            return None

    def plot_TSNE(self, metadata, features):
        targets = np.array([m.name for m in metadata])
        X_embedded = TSNE(n_components=2).fit_transform(features)

        for i, t in enumerate(set(targets)):
            idx = targets == t
            plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)

        plt.legend(bbox_to_anchor=(1, 1))
        plt.show()

    def find_threshold(self, db_metadata, db_features, plot=None):
        distances = []  # squared L2 distance between pairs
        identical = []  # 1 if same identity, 0 otherwise

        for i in range(len(db_metadata) - 1):
            for j in range(i + 1, len(db_metadata)):
                distances.append(self.distance(db_features[i], db_features[j]))
                identical.append(1 if db_metadata[i].name == db_metadata[j].name else 0)

        distances = np.array(distances)
        identical = np.array(identical)

        thresholds = np.arange(0.3, 1.0, 0.01)

        f1_scores = [f1_score(identical, distances < t) for t in thresholds]
        acc_scores = [accuracy_score(identical, distances < t) for t in thresholds]

        opt_idx = np.argmax(f1_scores)
        opt_tau = thresholds[opt_idx]  # Threshold at maximal F1 score
        opt_acc = accuracy_score(identical, distances < opt_tau)  # Accuracy at maximal F1 score
        print('Accuracy {0:1.2f} at threshold {1:1.2f}'.format(opt_acc, opt_tau))

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

            plt.figure(figsize=(12, 4))
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

        return opt_tau

    def train_classifier(self, db_metadata, db_features):
        targets = np.array([m.name for m in db_metadata])

        self.classifier_encoder = LabelEncoder()
        self.classifier_encoder.fit(targets)

        y = self.classifier_encoder.transform(targets)  # Numerical encoding of identities

        # Test/Train ratio 50/50
        train_idx = np.arange(db_metadata.shape[0]) % 2 != 0
        test_idx = np.arange(db_metadata.shape[0]) % 2 == 0

        X_train = db_features[train_idx]
        X_test = db_features[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        self.SVC = LinearSVC()
        self.SVC.fit(X_train, y_train)
        acc_svc = accuracy_score(y_test, self.SVC.predict(X_test))

        self.KNN = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
        self.KNN.fit(X_train, y_train)
        acc_knn = accuracy_score(y_test, self.KNN.predict(X_test))

        metadata_root = (db_metadata[0].base).split('/')[-1]
        print('KNN accuracy = {0:1.4f}, SVM accuracy = {1:1.4f} on {2}'.format(acc_knn, acc_svc, metadata_root))
        if acc_svc < 0.6 and acc_knn < 0.6:
            print('SVM and KNN accuracies are too low. Classifier disabled. \n "Lowest Score" identification enabled.')
        elif acc_svc > acc_knn:
            g_KNN_or_SVC = 1
            print('SVM classifier is activated.')
            self.classifier_valid = True
        else:
            g_KNN_or_SVC = 0
            print('KNN classification is enabled.')
            self.classifier_valid = True

    def predict_single_id(self, tg_metadata, tg_feature, threshold, plot=None):
        '''
            1. Predict a single target entry from database
            2. Compute average distance from target entry to predicted database identity
            3. Compare average distance to threshold (with some uncertaincy margin)
                Return either identity name or 'Unrecognised'.
        '''
        img = self.load_img(tg_metadata.img_path())
        if g_KNN_or_SVC == 0:
            ML = self.KNN
        elif g_KNN_or_SVC == 1:
            ML = self.SVC
        else:
            raise ValueError("Wrong 'g_KNN_or_SVC' value")

        prediction = ML.predict(tg_feature)  # returns metadata ID
        identity = self.classifier_encoder.inverse_transform(prediction)[0]

        avg_dist = self.avg_dist_tg_to_identity(tg_metadata, tg_feature, identity)
        if avg_dist < threshold + g_THRESHOLD_UNCERTAINTY:
            # print('Target {0} recognised {1}'.format(metadata.file, identity))
            if plot:
                plt.imshow(img)
                plt.title('recognised as {0}'.format(identity))
                plt.show()
            return identity, avg_dist
        else:
            # print('Target unrecognised')
            return 'Unrecognised', 0

    def avg_dist_tg_to_identity(self, tg_metadata, tg_feature, db_person_name):
        '''
            1. Computes distance from one TARGET entry to all DATABASE entries of a Specific person(label).
                                             img1    img2   img3  img4 (Database img)
                Example result: distances = [0.8 ,   0.5,   0.3,  1.1]
            2. Removes outliers in distances list with specificed upper/lower standard deviation (std)
                Example result: [0.8, 0.5]
            3. Computes Average value of all remaining distances.
                Example result: 0.75

            return: avg_dist -> Average distance from a single target img to a database identity.
                Example: jesper.jpg -> 0.56 distance to Jesper_Bro identitfy (8 imgs).
        '''
        lower_std = g_TARGET_TO_DB_NAME_lowerSTD
        upper_std = g_TARGET_TO_DB_NAME_upperSTD

        identify_db_paths = []
        identify_db_features = []

        for idx, item in enumerate(self.db_metadata):
            if item.name == db_person_name:
                identify_db_paths.append(item)
                identify_db_features.append(self.db_features[idx])

        all_dists = []
        for idx, item in enumerate(identify_db_paths):
            dist = self.distance(tg_feature, identify_db_features[idx])
            all_dists.append(dist)
            # print('File {0} has {1:1.4f} distance with {2}'.format(
            #     tg_metadata.file, dist, os.path.join(item.name, item.file)))

        if len(all_dists) > 1:
            all_dists_trimmed = []
            while len(all_dists_trimmed) == 0:
                all_dists_trimmed = trim_list_std(all_dists, lower_std, upper_std)
                lower_std += 0.01
                upper_std += 0.01

            avg_dist = sum(all_dists_trimmed) / len(all_dists_trimmed)
        else:
            avg_dist = sum(all_dists) / len(all_dists)

        return avg_dist

    def compute_identity(self, tg_metadata, tg_features, threshold):
        '''
            1. Predicts every entry in target metadata
            2. Builds dictionary with recognised identities, frequency and average distance.
            3. Selects the most fitting identity. Frequency > Average Distance.
        '''
        identity_list = []
        avg_dist_list = []
        features_reshaped = [item.reshape(1, -1) for item in tg_features]

        for x in range(len(features_reshaped)):
            identity, avg_dist = self.predict_single_id(tg_metadata[x], features_reshaped[x], threshold)
            identity_list.append(identity)
            avg_dist_list.append(avg_dist)

        identity_list, avg_dist_list = zip(*sorted(zip(identity_list, avg_dist_list)))  # Sort two lists in sync
        identity_list, avg_dist_list = (list(t) for t in zip(*sorted(zip(identity_list, avg_dist_list))))

        # ====== Getting frequency & average distance for each identity that was recognised
        result_dict = {}
        for idx, item in enumerate(identity_list):
            if item not in result_dict.keys():
                result_dict[item] = []

            if item in result_dict.keys():
                result_dict[item].append(avg_dist_list[idx])

        for item in result_dict:
            result_dict[item] = [len(result_dict[item]), (sum(result_dict[item]) / len(result_dict[item]))]
        # =========
        # CASE 1: If nothing has been recognised (only 'Unrecognised' key in dict)
        if len(result_dict) == 1 and 'Unrecognised' in result_dict:
            final_name = 'Unrecognised'
            final_min_dist = 0
            return final_name, final_min_dist
        else:
            #  CASE 2: If dict contains identities AND 'Unrecognised' -> process with identities.
            if 'Unrecognised' in result_dict:

                dict_sum = 0
                largest_value = 0
                for item in result_dict:
                    dict_sum += result_dict[item][0]
                unrec_ratio = result_dict['Unrecognised'][0] / dict_sum
                if unrec_ratio >= g_UNRECOGNISED_RATIO:
                    final_name = 'Unrecognised'
                    final_min_dist = 0
                    return final_name, final_min_dist
                else:
                    result_dict.pop('Unrecognised')

            largest_count = 0
            for item in result_dict:
                #  CASE 3: If dict contains more than 1 recognised target -> pick one with more photos recognised.
                if result_dict[item][0] > largest_count:
                    largest_count = result_dict[item][0]
                    final_name = item
                    final_min_dist = result_dict[item][1]
                #  CASE 4: If dict contains more than 1 recognised target with SAME number
                #  of recognised photos -> pick smallest avg dist
                elif result_dict[item][0] == largest_count:
                    print('whoops, {0} and {1} have the same number of recognised photos. \
                           Take smallest avg distance.'.format(item, final_name))
                    min_avg_dist = min(result_dict[item][1], result_dict[final_name][1])
                    if min_avg_dist == result_dict[item][1]:
                        largest_count = result_dict[item][0]
                        final_name = item
                        final_min_dist = result_dict[item][1]
                    else:
                        largest_count = result_dict[final_name][0]
        return final_name, final_min_dist
