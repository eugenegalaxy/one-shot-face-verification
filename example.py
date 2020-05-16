#!/usr/bin/env python3

import sys
import os
import time
# # Uncomment if have problems with python 2 vs python 3 missunderstanding.
# if '/opt/ros/melodic/lib/python2.7/dist-packages' in sys.path:
#     sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
# elif '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
#     sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
time_import_start = time.time()
from face_recognition.face_verification import FaceVerification
time_import_stop = time.time()

PRINT_EXECUTION_TIMES = False

def verify_target(database, target, img_mode):
    # STEP 1: Instantiate class object.
    time_FV = time.time()
    FV = FaceVerification()
    time_FV_stop = time.time()

    # STEP 2: Initilalise photo database
    time1 = time.time()
    FV.initDatabase(database)
    time1_stop = time.time()

    # STEP 3: Choose image acquisition mode (RECOMMENDED Mode 2)
    FV.setImgMode(img_mode)

    # STEP 4: Run the face verifier!
    time2 = time.time()
    if img_mode == 2:
        info, images = FV.predict(directory_path=target)
    elif img_mode == 1:
        info, images = FV.predict(single_img_path=target)
    else:
        assert ValueError('img_mode is not provided to verify_target')

    time2_stop = time.time()

    times = [0, 0, 0]
    times[0] = time_FV_stop - time_FV
    times[1] = time1_stop - time1
    times[2] = time2_stop - time2

    # info: Dictionary with information about the recognized person.
    # images: A list of image paths from the database of the person that was recognized ('just in case')
    return info, images, times


def print_target_info(target_info):
    # All keys that can be available:

    # This keys are ALWAYS available
    print('\n======================= Victim Information ==========================')
    if 'fullName' in target_info.keys():
        print('fullName:        {}'.format(target_info['fullName']))
    if 'nationality' in target_info.keys():
        print('nationality:     {}'.format(target_info['nationality']))
    if 'languageCode' in target_info.keys():
        print('languageCode:    {}'.format(target_info['languageCode']))
    if 'voiceRec' in target_info.keys():
        print('voiceRec:         {}'.format(target_info['voiceRec']))

    # This keys re available only if database is mysql_database
    if 'age' in target_info.keys():
        print('age:             {}'.format(target_info['age']))
    if 'weightKg' in target_info.keys():
        print('weightKg:        {}'.format(target_info['weightKg']))
    if 'heightCm' in target_info.keys():
        print('heightCm:        {}'.format(target_info['heightCm']))
    if 'socialMediaLink' in target_info.keys():
        print('socialMediaLink: {}'.format(target_info['socialMediaLink']))

if __name__ == "__main__":
    time_main_start = time.time()

    path = os.path.dirname(os.path.abspath(__file__))

    database_1 = path + '/face_recognition/images/manual_database'  # Option 1
    database_2 = path + '/face_recognition/images/mysql_database'   # Option 2

    dir_path_1 = path + '/face_recognition/images/new_entries/jevgenijs_galaktionovs'
    dir_path_2 = path + '/face_recognition/images/new_entries/jesper_bro'
    dir_path_3 = path + '/face_recognition/images/new_entries/lelde_skrode'
    dir_path_4 = path + '/face_recognition/images/new_entries/hugo_markoff'
    dir_path_5 = path + '/face_recognition/images/new_entries/arnold'

    ALL_FROM_DIRECTORY = 2
    SINGLE_IMAGE_PATH = 1

    target_info, target_images, times = verify_target(database_2, dir_path_5, ALL_FROM_DIRECTORY)
    print_target_info(target_info)

    time_main_stop = time.time()
    impo_time = time_import_stop - time_import_start
    main_time = time_main_stop - time_main_start

    if PRINT_EXECUTION_TIMES is True:
        print('\n========================= Execution Times ===========================')
        print('Module imports executed in {0:1.2f} seconds.'.format(impo_time))
        print('FV = FaceVerification() executed in {0:1.2f} seconds.'.format(times[0]))
        print('FV.initDatabase() executed in {0:1.2f} seconds.'.format(times[1]))
        print('FV.predict() executed in {0:1.2f} seconds.'.format(times[2]))
        print('Program executed in {0:1.2f} seconds.'.format(main_time))
        print('Total time is {0:1.2f} seconds.'.format(impo_time + main_time))
