import os
import sys
import mysql.connector
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Adding script to PATH to import 'directory_utils' from parent folder.
from directory_utils import load_metadata, load_metadata_short


def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        binaryData = file.read()
    return binaryData


def insert_employee_images(fullName, empImage, empId):
    print("Inserting row into emloyee_images table")
    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='ReallyARobot',
                                             user='root',
                                             #password='123456'
                                             )
        connection.set_charset_collation('utf8')  # Default utf-8 encoding fails to read BLOB(images) data
        cursor = connection.cursor()
        sql_query = """ INSERT INTO employee_images
                          (imgId, fullName, empImage, empId) VALUES (%s,%s,%s,%s)"""

        empPicture = convertToBinaryData(empImage)

        insert_tuple = (None, fullName, empPicture, empId)  # Convert data into tuple format
        cursor.execute(sql_query, insert_tuple)
        connection.commit()
        print("Insertion is successful")

    except mysql.connector.Error as error:
        print("Failed inserting BLOB data into MySQL table {}".format(error))

    finally:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print("MySQL connection is closed")


# # target_database folder -> 1x image of 17 people divided in directories
# img_paths = load_metadata('./face_recognition/images/manual_database', names=1)
# img_paths = [item.image_path() for item in img_paths]

# insert_employee_images("Iuliu Novak",               img_paths[0],  5)
# insert_employee_images("Hugo Markoff",              img_paths[1],  2)
# insert_employee_images("Ivelin Krasimirov Penchev", img_paths[2],  6)
# insert_employee_images("Jakob Høy",                 img_paths[3],  7)
# insert_employee_images("Jesper Bro Rosenberg",      img_paths[4], 18)
# insert_employee_images("Jevgenijs Galaktionovs",    img_paths[5],  1)
# insert_employee_images("Johnni Nielsen",            img_paths[6],  8)
# insert_employee_images("Lasma Ziedina",             img_paths[7],  9)
# insert_employee_images("Lelde Skrode",              img_paths[8],  4)
# insert_employee_images("Line Uggerly Jørgensen",    img_paths[9],  10)
# insert_employee_images("Mario Javier Rincón Pérez", img_paths[10],  11)
# insert_employee_images("Rebecca Nekou Malihi",      img_paths[11], 3)
# insert_employee_images("Steffan Svendsen",          img_paths[12], 12)
# insert_employee_images("Strahinja Dosen",           img_paths[13], 13)
# insert_employee_images("David Michalik",            img_paths[14], 14)
# insert_employee_images("Therese Haagen Søndergaard",img_paths[15], 15)
# insert_employee_images("Toke Olsen",                img_paths[16], 16)
# insert_employee_images("Uffe Koch",                 img_paths[17], 17)


# path = '/home/eugenegalaxy/Desktop/extra_images'
# img_paths = load_metadata(path, names=1)
# img_paths = [item.image_path() for item in img_paths]
# insert_employee_images("Hugo Markoff",img_paths[0],  2)
# insert_employee_images("Hugo Markoff",img_paths[1],  2)
# insert_employee_images("Hugo Markoff",img_paths[2],  2)
# insert_employee_images("Hugo Markoff",img_paths[3],  2)
# insert_employee_images("Hugo Markoff",img_paths[4],  2)
# insert_employee_images("Hugo Markoff",img_paths[5],  2)
# insert_employee_images("Hugo Markoff",img_paths[6],  2)
# insert_employee_images("Hugo Markoff",img_paths[7],  2)
# insert_employee_images("Hugo Markoff",img_paths[8],  2)
# insert_employee_images("Hugo Markoff",img_paths[9],  2)

# insert_employee_images("Jevgenijs Galaktionovs",    img_paths[10],  1)
# insert_employee_images("Jevgenijs Galaktionovs",    img_paths[11],  1)
# insert_employee_images("Jevgenijs Galaktionovs",    img_paths[12],  1)
# insert_employee_images("Jevgenijs Galaktionovs",    img_paths[13],  1)
# insert_employee_images("Jevgenijs Galaktionovs",    img_paths[14],  1)
# insert_employee_images("Jevgenijs Galaktionovs",    img_paths[15],  1)
# insert_employee_images("Jevgenijs Galaktionovs",    img_paths[16],  1)
# insert_employee_images("Jevgenijs Galaktionovs",    img_paths[17],  1)
# insert_employee_images("Jevgenijs Galaktionovs",    img_paths[18],  1)
# insert_employee_images("Jevgenijs Galaktionovs",    img_paths[19],  1)

# insert_employee_images("Lelde Skrode",              img_paths[20],  4)
# insert_employee_images("Lelde Skrode",              img_paths[21],  4)
# insert_employee_images("Lelde Skrode",              img_paths[22],  4)
# insert_employee_images("Lelde Skrode",              img_paths[23],  4)
# insert_employee_images("Lelde Skrode",              img_paths[24],  4)
# insert_employee_images("Lelde Skrode",              img_paths[25],  4)
# insert_employee_images("Lelde Skrode",              img_paths[26],  4)
# insert_employee_images("Lelde Skrode",              img_paths[27],  4)
# insert_employee_images("Lelde Skrode",              img_paths[28],  4)
# insert_employee_images("Lelde Skrode",              img_paths[29],  4)