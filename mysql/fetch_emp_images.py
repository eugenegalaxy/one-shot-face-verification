import mysql.connector


path = 'mysql_database'


def write_file(data, filename):
    # Convert binary data to proper format and write it on Hard Disk
    with open(filename, 'wb') as file:
        file.write(data)


def fetch_table(table_name):
    print('Reading data from {} table'.format(table_name))

    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='ReallyARobot',
                                             user='root',
                                             password='123456')

        connection.set_charset_collation('latin1')  #  Default utf-8 encoding fails to read BLOB(images) data    

        cursor = connection.cursor()
        query = '''SELECT * FROM {}'''.format(table_name)
        cursor.execute(query)
        data = cursor.fetchall()
        # for row in data:
            # print('Id = ', row[0], )
            # print('Name = ', row[1])
            # image = row[2]
            # print('emp_key') = row[3]
            # # print('Storing employee image and bio-data on disk \n')
            # write_file(image, photo)

        #                        DESCRIBE" Structure
        #           _______________________________________________________
        # Index:   |_____0______|_____1_____|__2___|__3__|___4_____|___5___|
        # Contents:| Field name | Data Type | Null | Key | Default | Extra |
        #          --------------------------------------------------------

        table_primary_key = []  # Or at least UNIQUE ID bcuz of "auto_increment". We just need that.
        table_foreign_key = []

        query = '''DESCRIBE {}'''.format(table_name)
        cursor.execute(query)
        data = cursor.fetchall()
        for row in data:
            print(row)
            for entry in row:
                if entry == 'PRI' or 'auto_increment':
                    table_primary_key.append(row[0])
                if entry == 'MUL':
                    table_foreign_key.append(row[0])
        print(table_primary_key)
        print(table_foreign_key)
    except mysql.connector.Error as error:
        print('Failed to read BLOB data from MySQL table {}'.format(error))

    finally:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print('MySQL connection is closed')

# readBLOB(1, 'D:\Python\Articles\my_SQL\query_output\eric_photo.png',
#          'D:\Python\Articles\my_SQL\query_output\eric_bioData.txt')
# readBLOB(2, 'D:\Python\Articles\my_SQL\query_output\scott_photo.png',
#          'D:\Python\Articles\my_SQL\query_output\scott_bioData.txt')

tableName = 'employee_images'
# tableName = 'employee_profiles'
fetch_table(tableName)
