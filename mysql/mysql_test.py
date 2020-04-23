# -----------------------------------------------------------------------------------------
# Code taken from https://www.mysqltutorial.org/python-connecting-mysql-databases/
# -----------------------------------------------------------------------------------------
from mysql.connector import MySQLConnection, Error
from python_mysql_dbconfig import read_db_config
import base64
import io
import cv2
from directory_utils import load_metadata
import PIL.Image


def connect():
    """ Connect to MySQL database """

    db_config = read_db_config()
    conn = None
    try:
        print('Connecting to MySQL database...')
        conn = MySQLConnection(**db_config)

        if conn.is_connected():
            print('Connection established.')
        else:
            print('Connection failed.')

    except Error as error:
        print(error)

    return conn


def disconnect(connector_obj):
    if connector_obj is not None and connector_obj.is_connected():
        connector_obj.close()
        print('Connection closed.')


img_paths = load_metadata('target_database', 1)

img = [0] * len(img_paths)
encodestring = [0] * len(img_paths)

for i, path in enumerate(img_paths):
    img[i] = cv2.imread(str(path), 1)
    encodestring[i] = base64.b64encode(img[i])


def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        binaryData = file.read()
    return binaryData


db = connect()
mycursor = db.cursor()

sql = "INSERT INTO employee_images(imgId,firstName,lastName,empImage,empId) VALUES (%s, %s, %s, %s, %s)"
val = (None, 'Lelde', 'Skrode', encodestring[8], 1)
# # val = [
# #   ('Peter', 'Lowstreet 4'),
# #   ('Viola', 'Sideway 1633')
# # ]
mycursor.execute(sql, val)

# # sql = "INSERT INTO sample values(%s)"
# # mycursor.execute(sql, (encodestring,))
# # mycursor.executemany(sql, val) #  For multiple-row input
db.commit()
# print(mycursor.rowcount, "was inserted.")

sql1 = "SELECT empImage FROM employee_images WHERE imgId = 4"
mycursor.execute(sql1)
data = mycursor.fetchall()

data1 = base64.b64decode(data[0][0])

file_like = io.BytesIO(data1)
print(type(file_like))
# file_bytes = np.asarray(bytearray(file_like.read()), dtype=np.uint8)
# img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
# print(file_like)
# io_buf = io.BytesIO(buffer)
img = PIL.Image.open(file_like)
print(type(img))
img.show()
# decode_img = cv2.imdecode(np.fromfile(file_like.read(), np.uint8), 1)
# cv2.imshow('RealSense', decode_img)
# cv2.waitKey(0)


disconnect(db)
