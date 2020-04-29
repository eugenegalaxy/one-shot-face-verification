# Libraries to classify identity
# from sklearn.preprocessing import LabelEncoder
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import LinearSVC
# from sklearn.manifold import TSNE  # To plot the results of classifying
# from sklearn.metrics import f1_score, accuracy_score  # Evaluation

def verify_target():
    # # STEP 1: Instantiate class object.
    # FV = FaceVerification()

    # # STEP 2: Initilalise photo database (images to compare new entries to)
    # database_1 = 'images/manual_database'  # Option 1
    # database_2 = 'images/mysql_database'   # Option 2
    # FV.initDatabase(database_2, target_names=1)

    # # STEP 3: Choose image acquisition mode (RECOMMENDED Mode 3)

    # GET_FRESH_CAM_IMAGE = 0  # Image Mode 1 -> Acquire fresh image and compare it against database.
    # FV.setImgMode(GET_FRESH_CAM_IMAGE)

    # SINGLE_IMAGE_PATH = 1  # Image Mode 2 -> Provide path to image saved on disk and verify it against database.
    # FV.setImgMode(SINGLE_IMAGE_PATH()
    # my_image = 'images/new_entries/image_0000.jpg'  # EXAMPLE PATH, MIGHT NOT EXIST

    # ALL_FROM_DIRECTORY = 2  # Image Mode 3 (RECOMMENDED) -> Provide path to directory and scan ALL images in it (NOTE: No subdirectories!)
    # FV.setImgMode(ALL_FROM_DIRECTORY)
    # # Options for Image Mode 3:
    # dir_path_1 = 'images/new_entries/jevgenijs_galaktionovs'
    # dir_path_2 = 'images/new_entries/jesper_bro'
    # dir_path_3 = 'images/new_entries/lelde_skrode'
    # dir_path_4 = 'images/new_entries/hugo_markoff'
    # dir_path_5 = 'images/new_entries/arnold'

    # # STEP 4: Run the machine and get data!
    # info, images = FV.predict(plot=1)  # Example for Image Mode 1
    # info, images = FV.predict(single_img_path=my_image)  # Example for Image Mode 2
    # info, images = FV.predict(directory_path=dir_path_1)  # Example for Image Mode 3
    # return info, images
    pass


def plot_pair_distance_example(metadata, embedded, idx1, idx2):

    # dist = distance(embedded[idx1], embedded[idx2])

    # plt.figure(figsize=(8, 3))
    # plt.suptitle("Distance = {0:1.3f}".format(dist))
    # plt.subplot(121)
    # plt.imshow(load_image(metadata[idx1].image_path()))
    # plt.subplot(122)
    # plt.imshow(load_image(metadata[idx2].image_path()))
    # plt.show()
    pass


def evaluatate_training_data(metadata, embedded):
    # distances = []  # squared L2 distance between pairs
    # identical = []  # 1 if same identity, 0 otherwise

    # num = len(metadata)

    # for i in range(num - 1):
    #     for j in range(i + 1, num):
    #         distances.append(distance(embedded[i], embedded[j]))
    #         identical.append(1 if metadata[i].name == metadata[j].name else 0)

    # distances = np.array(distances)
    # identical = np.array(identical)

    # thresholds = np.arange(0.3, 1.0, 0.01)

    # f1_scores = [f1_score(identical, distances < t) for t in thresholds]
    # acc_scores = [accuracy_score(identical, distances < t) for t in thresholds]

    # opt_idx = np.argmax(f1_scores)
    # # Threshold at maximal F1 score
    # opt_tau = thresholds[opt_idx]
    # # Accuracy at maximal F1 score
    # opt_acc = accuracy_score(identical, distances < opt_tau)

    # # Plot F1 score and accuracy as function of distance threshold
    # plt.plot(thresholds, f1_scores, label='F1 score')
    # plt.plot(thresholds, acc_scores, label='Accuracy')
    # plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
    # plt.title('Accuracy at threshold {0:1.2f} = {1:1.3f}'.format(opt_tau, opt_acc))
    # plt.xlabel('Distance threshold')
    # plt.legend()
    # plt.show()

    # return opt_tau
    pass


def plot_histograms(metadata, embedded, threshold):
    # distances = []  # squared L2 distance between pairs
    # identical = []  # 1 if same identity, 0 otherwise

    # num = len(metadata)

    # for i in range(num - 1):
    #     for j in range(i + 1, num):
    #         distances.append(distance(embedded[i], embedded[j]))
    #         identical.append(1 if metadata[i].name == metadata[j].name else 0)

    # distances = np.array(distances)
    # identical = np.array(identical)

    # dist_pos = distances[identical == 1]
    # dist_neg = distances[identical == 0]

    # plt.figure(figsize=(12, 4))

    # plt.subplot(121)
    # plt.hist(dist_pos)
    # plt.axvline(x=threshold, linestyle='--', lw=1, c='lightgrey', label='Threshold')
    # plt.title('Distances (pos. pairs)')
    # plt.legend()

    # plt.subplot(122)
    # plt.hist(dist_neg)
    # plt.axvline(x=threshold, linestyle='--', lw=1, c='lightgrey', label='Threshold')
    # plt.title('Distances (neg. pairs)')
    # plt.legend()
    # plt.show()
    pass


def init_classifier(metadata, embedded):
    # targets = np.array([m.name for m in metadata])

    # encoder = LabelEncoder()
    # encoder.fit(targets)

    # # Numerical encoding of identities
    # y = encoder.transform(targets)

    # train_idx = np.arange(metadata.shape[0]) % 2 != 0
    # print("train_idx: ", train_idx)
    # test_idx = np.arange(metadata.shape[0]) % 2 == 0

    # # 50 train examples of 10 identities (5 examples each)
    # X_train = embedded[train_idx]
    # # 50 test examples of 10 identities (5 examples each)
    # X_test = embedded[test_idx]

    # y_train = y[train_idx]
    # y_test = y[test_idx]

    # knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    # svc = LinearSVC()

    # knn.fit(X_train, y_train)
    # svc.fit(X_train, y_train)

    # acc_knn = accuracy_score(y_test, knn.predict(X_test))
    # acc_svc = accuracy_score(y_test, svc.predict(X_test))

    # print('KNN accuracy = {0}, SVM accuracy = {1}'.format(acc_knn, acc_svc))
    # return test_idx, svc, knn, encoder
    pass


def DoPredict_example(idx):
    # example_image = load_image(database[test_idx][idx].image_path())
    # example_prediction = svc.predict([emb_database[test_idx][idx]])
    # example_identity = encoder.inverse_transform(example_prediction)[0]

    # plt.imshow(example_image)
    # plt.title('Recognized as {0}'.format(example_identity))
    # plt.show()
    pass


def plot_classifying_results(metadata):
    # targets = np.array([m.name for m in metadata])
    # X_embedded = TSNE(n_components=2).fit_transform(emb_database)

    # for i, t in enumerate(set(targets)):
    #     idx = targets == t
    #     plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)

    # plt.legend(bbox_to_anchor=(1, 1))
    # plt.show()
    pass


def comparing_images_scores_TEST():
    # img1 = get_webcam_image(save=1)  # perevernutaja
    # print('img1 Dimensions :', img1.shape)
    # img1_emb = FV.get_features_img(img1)
    # time.sleep(5)

    # img2 = cv2.imread('new_entries/person/image.jpg', 1)  # ne perevernutaja
    # print('img2 Dimensions :', img2.shape)
    # img2_emb = FV.get_features_img(img2)

    # img3 = img2[..., ::-1]  # perevernutaja
    # print('img3 Dimensions :', img3.shape)
    # img3_emb = FV.get_features_img(img3)

    # test_img = cv2.imread('target_database/Jevgenijs_Galaktionovs-ru/ae.jpg', 1)
    # print('test_img Dimensions :', test_img.shape)
    # test_img_emb = FV.get_features_img(test_img)

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
    pass


# opt_threshold = evaluatate_training_data(database, emb_database)
# plot_histograms(database, emb_database, opt_threshold)
# test_idx, svc, knn, encoder = init_classifier(database, emb_database)
