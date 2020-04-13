import os
import sys
import mysql.connector
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Adding script to PATH to import 'directory_utils' from parent folder.
from directory_utils import load_metadata


def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        binaryData = file.read()
    return binaryData


def insert_employee_images(firstName, lastName, empImage, empId):
    print("Inserting row into emloyee_images table")
    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='ReallyARobot',
                                             user='root',
                                             password='123456')
        cursor = connection.cursor()
        sql_query = """ INSERT INTO employee_images
                          (imgId, firstName, lastName, empImage, empId) VALUES (%s,%s,%s,%s,%s)"""

        empPicture = convertToBinaryData(empImage)

        insert_tuple = (None, firstName, lastName, empPicture, empId)  # Convert data into tuple format
        cursor.execute(sql_query, insert_tuple)
        connection.commit()
        print("Image and file inserted successfully as a BLOB into python_employee table")

    except mysql.connector.Error as error:
        print("Failed inserting BLOB data into MySQL table {}".format(error))

    finally:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print("MySQL connection is closed")


img_paths = load_metadata('target_database')
img_paths = [str(item) for item in img_paths]
insert_employee_images("Jevgenijs", "Galaktionovs", img_paths[4], 1)
