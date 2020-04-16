import mysql.connector

ENCODING = 'latin1'  # Default utf-8 encoding fails to read BLOB(images) data
path = 'mysql_database'  # To save images from database.

# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# FOR DATABASE USERS, PLEASE WRITE IN THE NAMES:

# Host address:
g_host_address = 'localhost'
# Database name
g_database_name = 'ReallyARobot'
# Username:
g_username = 'root'
# Password:
g_password = '123456'
# 1. Database table containing employee information (name, age, nationality, etc)
g_emp_prof = 'employee_profiles'
# 2. Database table containing employee images (Must be of BLOB type) TODO -> not only BLOBs, but also references
g_emp_images = 'employee_images'
# 3. Database table column with employee names (ASSUMED TO BE IN table 1.) Used for verification
g_emp_name = 'fullName'
# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------


def save_image_on_disk(data, filename):
    # Convert binary data to proper format and write it on Hard Disk
    if not isinstance(data, bytes):
        data = data.encode(ENCODING)
    with open(filename, 'wb') as file:
        file.write(data)


def fetch_table_description(table_name):
    print('Reading data from {} table'.format(table_name))

    try:
        connection = mysql.connector.connect(host=g_host_address,
                                             database=g_database_name,
                                             user=g_username,
                                             password=g_password)

        connection.set_charset_collation(ENCODING)
        cursor = connection.cursor()
        query = '''DESCRIBE {}'''.format(table_name)
        cursor.execute(query)
        data = cursor.fetchall()

        description_dic = {
            'Table_name': table_name,
            'Primary_key_column': list(),
            'Foreign_key_column': list(),
            'BLOB_data_column': list(),
        }

        for idx, row in enumerate(data):
            description_dic['Row_{}'.format(idx)] = row
            for entry in row:
                if entry == 'PRI':
                    description_dic['Primary_key_column'].append([row[0], idx])
                if entry == 'MUL':
                    description_dic['Foreign_key_column'].append([row[0], idx])
                if entry == 'longblob' or entry == 'mediumblob' or entry == 'blob':
                    description_dic['BLOB_data_column'].append([row[0], idx])
        return description_dic

    except mysql.connector.Error as error:
        print('Failed to read BLOB data from MySQL table {}'.format(error))

    finally:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print('MySQL connection is closed')


def fetch_table(table_name):
    print('Reading data from {} table'.format(table_name))

    try:
        connection = mysql.connector.connect(host=g_host_address,
                                             database=g_database_name,
                                             user=g_username,
                                             password=g_password)

        connection.set_charset_collation('latin1')  # Default utf-8 encoding fails to read BLOB(images) data

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
        return data

    except mysql.connector.Error as error:
        print('Failed to read BLOB data from MySQL table {}'.format(error))

    finally:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print('MySQL connection is closed')


# Function assumes that table 'g_emp_images' is connected to table g_emp_prof by FOREIGN KEY
def combine_data_by_foreign_key():
    try:
        connection = mysql.connector.connect(host=g_host_address,
                                             database=g_database_name,
                                             user=g_username,
                                             password=g_password)

        connection.set_charset_collation(ENCODING)
        cursor = connection.cursor()

        query = '''
                    SELECT referenced_column_name, table_name, column_name
                    FROM information_schema.KEY_COLUMN_USAGE
                    WHERE table_schema = %s
                    AND referenced_table_name = %s;
                '''

        args = (g_database_name, g_emp_prof)
        cursor.execute(query, args)

        refer_info = cursor.fetchall()  # [(parent_column_name, child_table_name, child_column_name)]
        if refer_info:
            parent_column_name = refer_info[0][0]
            child_table_name = refer_info[0][1]
            child_column_name = refer_info[0][2]

            child_table_desc = fetch_table_description(child_table_name)
            if child_table_desc['BLOB_data_column']:
                BLOB_column_name = child_table_desc['BLOB_data_column'][0][0]   # Shows name of table column that has BLOB object
                print(BLOB_column_name)

                query = '''
                            SELECT p.{0}, i.{1} FROM {2} p
                            INNER JOIN {3} i
                            ON i.{4} = p.{5};
                        '''.format(g_emp_name, BLOB_column_name, g_emp_prof, g_emp_images, child_column_name, parent_column_name)

                cursor.execute(query)
                merged_data = cursor.fetchall()

# ------------------------------------------------------------------------
# Stopped here. Insted of this block, replace it with cool sorter by folders/names! Also collect all info into .txt file.
                counter = 0
                funny_name = 'ae'
                ext = 'jpg'

                for item in merged_data:
                    full_path = "{0}/{1}_{2}.{3}".format(path, funny_name, str(counter), ext)
                    save_image_on_disk(item[1], full_path)
                    counter += 1
# ------------------------------------------------------------------------
            else:
                print('There are not images in the table.')
        else:
            print('Two tables are not connected by any FOREIGN KEY')
    except mysql.connector.Error as error:
        print('Failed to read BLOB data from MySQL table {}'.format(error))

    finally:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print('MySQL connection is closed')    
    # Function that checks two tables for FOREIGN KEY and collects data into 1 table


combine_data_by_foreign_key()

# if table_desc['Primary_key_column']:
#     PK_column_id = table_desc['Primary_key_column'][0][1]  # Shows index of table column that has PRIMARY KEY
# if table_desc['Foreign_key_column']:
#     FK_column_id = table_desc['Foreign_key_column'][0][1]  # Shows index of table column that has FOREIGN KEY
# if table_desc['BLOB_data_column']:
#     BLOB_column_id = table_desc['BLOB_data_column'][0][1]   # Shows index of table column that has BLOB object

# person_name = table_data[0][1]
# person_name.replace(" ", "+")
# print(person_name)
# path = 'mysql_database'
# funny_name = 'iuliu.jpg'
# full_path = "{0}/{1}".format(path, funny_name)

# image = table_data[0][BLOB_column_id]
# print(type(image))
# save_image_on_disk(image, full_path)




emp_prof_data = fetch_table(g_emp_prof)
emp_images_data = fetch_table(g_emp_images)

emp_prof_desc = fetch_table_description(g_emp_prof)
emp_images_desc = fetch_table_description(g_emp_images)