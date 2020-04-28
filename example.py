#!/usr/bin/env python3
from face_verification import FaceVerification


def verify_target():
    # STEP 1: Instantiate class object.
    FV = FaceVerification()

    # STEP 2: Initilalise photo database (images to compare new entries to)
    database_1 = 'images/manual_database'  # Option 1
    database_2 = 'images/mysql_database'   # Option 2
    FV.initDatabase(database_2, target_names=1)

    # STEP 3: Choose image acquisition mode (RECOMMENDED Mode 2)
    ALL_FROM_DIRECTORY = 2  # Provide path to directory and scan ALL images in it (NOTE: No subdirectories!)
    FV.setImgMode(ALL_FROM_DIRECTORY)

    # STEP 4: Specify folder where NEWly captured images are and run the face verifier!
    # Options for Rebecca (not for general purpose)
    dir_path_1 = 'images/new_entries/jevgenijs_galaktionovs'
    dir_path_2 = 'images/new_entries/jesper_bro'
    dir_path_3 = 'images/new_entries/lelde_skrode'
    dir_path_4 = 'images/new_entries/hugo_markoff'
    dir_path_5 = 'images/new_entries/arnold'

    info, images = FV.predict(directory_path=dir_path_1, plot=1)

    return info, images


if __name__ == "__main__":
    target_info, target_images = verify_target()
    # target_info: Dictionary with information about the recognized person.
    #   To access data, use target_info[name_of_the_key] where name_of_the_key are keys. Example target_info['fullName']
    #   Note: target_info will always contain 'fullName', 'languageCode' and 'voiceRec'. So you can always take it.
    # target_images: A list of image paths from the database of the person that was recognized ('just in case')
    print("\n".join("{}\t{}".format(k, v) for k, v in target_info.items()))  # Just to print dictionary line by line
