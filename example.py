#!/usr/bin/env python3

import sys
import os
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from face_recognition.face_verification import FaceVerification


def verify_target():
    # STEP 1: Instantiate class object.
    FV = FaceVerification()
    path = os.path.dirname(os.path.abspath(__file__))
    print(path)
    # STEP 2: Initilalise photo database (images to compare new entries to)
    database_1 = path + '/face_recognition/images/manual_database'  # Option 1
    database_2 = path + '/face_recognition/images/mysql_database'   # Option 2
    FV.initDatabase(database_2, target_names=1)

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

    info, images = FV.predict(directory_path=dir_path_1, plot=1)

    return info, images


if __name__ == "__main__":
    target_info, target_images = verify_target()
    # target_info: Dictionary with information about the recognized person.
    # target_images: A list of image paths from the database of the person that was recognized ('just in case')

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
