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


def verify_target():
    # STEP 1: Instantiate class object.
    time_FV = time.time()
    FV = FaceVerification()
    print('FV = FaceVerification() executed in {} seconds.'.format(time.time() - time_FV))
    path = os.path.dirname(os.path.abspath(__file__))
    # STEP 2: Initilalise photo database (images to compare new entries to)
    database_1 = path + '/face_recognition/images/manual_database'  # Option 1
    database_2 = path + '/face_recognition/images/mysql_database'   # Option 2
    time1 = time.time()
    FV.initDatabase(database_2, target_names=1)
    print('FV.initDatabase() executed in {} seconds.'.format(time.time() - time1))
    # STEP 3: Choose image acquisition mode (RECOMMENDED Mode 2)
    ALL_FROM_DIRECTORY = 2  # Provide path to directory and scan ALL images in it (NOTE: No subdirectories!)
    FV.setImgMode(ALL_FROM_DIRECTORY)

    # STEP 4: Specify folder where NEWly captured images are and run the face verifier!
    # Options for Rebecca (not for general purpose)
    dir_path_1 = path + '/face_recognition/images/new_entries/jevgenijs_galaktionovs'
    dir_path_2 = path + '/face_recognition/images/new_entries/jesper_bro'
    dir_path_3 = path + '/face_recognition/images/new_entries/lelde_skrode'
    dir_path_4 = path + '/face_recognition/images/new_entries/hugo_markoff'
    dir_path_5 = path + '/face_recognition/images/new_entries/arnold'

    time2 = time.time()
    info, images = FV.predict(directory_path=dir_path_1)
    print('FV.predict() executed in {} seconds.'.format(time.time() - time2))
    # info: Dictionary with information about the recognized person.
    # images: A list of image paths from the database of the person that was recognized ('just in case')
    return info, images


def print_target_info(target_info):
    # All keys that can be available:

    # This keys are ALWAYS available
    if 'fullName' in target_info.keys():
        print('fullName is {}'.format(target_info['fullName']))
    if 'languageCode' in target_info.keys():
        print('languageCode is {}'.format(target_info['languageCode']))
    if 'voiceRec' in target_info.keys():
        print('voiceRec is {}'.format(target_info['voiceRec']))

    # This keys re available only if database is mysql_database
    if 'weightKg' in target_info.keys():
        print('weightKg is {}'.format(target_info['weightKg']))
    if 'age' in target_info.keys():
        print('age is {}'.format(target_info['age']))
    if 'heightCm' in target_info.keys():
        print('heightCm is {}'.format(target_info['heightCm']))
    if 'socialMediaLink' in target_info.keys():
        print('socialMediaLink is {}'.format(target_info['socialMediaLink']))
    if 'nationality' in target_info.keys():
        print('nationality is {}'.format(target_info['nationality']))
    if 'empId' in target_info.keys():
        print('empId is {}'.format(target_info['empId']))

if __name__ == "__main__":
    time_main_start = time.time()

    target_info, target_images = verify_target()
    print_target_info(target_info)

    time_main_stop = time.time()
    impo_time = time_import_stop - time_import_start
    main_time = time_main_stop - time_main_start
    print('\nModule imports executed in {} seconds.'.format(impo_time))
    print('Program executed in {} seconds.'.format(main_time))
    print('Total time is {} seconds.'.format(impo_time + main_time))
