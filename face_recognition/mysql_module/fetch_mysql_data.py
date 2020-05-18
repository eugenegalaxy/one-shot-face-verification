import mysql.connector
import os

ENCODING = 'latin1'  # Default utf-8 encoding fails to read BLOB(images) data

# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# ============================  FOR DATABASE USERS, PLEASE WRITE IN THE INFORMATION ============================

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
        file.close()


def save_data_text_on_disk(data, filename):
    with open(filename, "a") as file:
        file.write(str(data))
        file.close()


def query_database(query, args=None):
    try:
        connection = mysql.connector.connect(host=g_host_address,
                                             database=g_database_name,
                                             user=g_username,
                                             #password=g_password
                                             )

        connection.set_charset_collation(ENCODING)  # Default utf-8 encoding fails to read BLOB(images) data

        cursor = connection.cursor()
        if args is not None:
            cursor.execute(query, args)
        else:
            cursor.execute(query)

        data = cursor.fetchall()
        return data

    except mysql.connector.Error as error:
        print(error)

    finally:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print('MySQL connection is closed')


def fetch_table_description(table_name):
    print('Reading data from {} table'.format(table_name))

    query = '''DESCRIBE {}'''.format(table_name)
    data = query_database(query)

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


# Function assumes that table 'g_emp_images' is connected to table g_emp_prof by FOREIGN KEY
def save_employee_data(save_path):
    '''Function combines two tables data by foreign key and saves data on disk.'''
    # ============================= TEXT PART ===================================
    query = '''
                SELECT `COLUMN_NAME`
                FROM `INFORMATION_SCHEMA`.`COLUMNS`
                WHERE `TABLE_SCHEMA`= %s
                AND `TABLE_NAME`= %s;
            '''
    args = (g_database_name, g_emp_prof)
    column_fields = query_database(query, args)
    column_names = [val for sublist in column_fields for val in sublist]

    query = '''SELECT * FROM {}'''.format('employee_profiles')
    all_entries = query_database(query)

    # ============================= IMAGES PART =======================================
    query = '''
                SELECT referenced_column_name, table_name, column_name
                FROM information_schema.KEY_COLUMN_USAGE
                WHERE table_schema = %s
                AND referenced_table_name = %s;
            '''
    args = (g_database_name, g_emp_prof)
    refer_info = query_database(query, args)  # [(parent_column_name, child_table_name, child_column_name)]

    if refer_info:
        parent_column_name = refer_info[0][0]
        child_table_name = refer_info[0][1]
        child_column_name = refer_info[0][2]

        child_table_desc = fetch_table_description(child_table_name)

        if child_table_desc['BLOB_data_column']:
            BLOB_column_name = child_table_desc['BLOB_data_column'][0][0]   # Shows name of table column that has BLOB object

            query = '''
                        SELECT p.{0}, i.{1} FROM {2} p
                        INNER JOIN {3} i
                        ON i.{4} = p.{5};
                    '''.format(g_emp_name, BLOB_column_name, g_emp_prof, g_emp_images, child_column_name, parent_column_name)
            merged_data = query_database(query)

    # ============================= SAVING TO DISK PART =======================================
            for item in merged_data:
                counter = 0  # TODO replace this filename numerator with already existing from directory_utils!
                person_name = item[0].replace(" ", "_")
                directory_path = os.path.join(save_path, person_name)
                if not os.path.isdir(directory_path):
                    os.mkdir(directory_path)

                full_path_img = "{0}/{1}_{2}.{3}".format(directory_path, person_name, str(counter), 'jpg')
                while os.path.isfile(full_path_img):  # if image exists with same filename, iterate counter (Example "image_0.jpg -> image_1.jpg")
                    counter += 1
                    full_path_img = "{0}/{1}_{2}.{3}".format(directory_path, person_name, str(counter), 'jpg')

                save_image_on_disk(item[1], full_path_img)

                full_path_txt = "{0}/{1}.{2}".format(directory_path, 'info', 'txt')
                if not os.path.isfile(full_path_txt):  # If info txt already exists, it won't save the same info on every photo iteration
                    for entry in all_entries:
                        if item[0] in entry:
                            for idx, row in enumerate(entry):  # <-- iterates over each entry
                                data_str = "{0}: {1}\n".format(column_names[idx], row)
                                save_data_text_on_disk(data_str, full_path_txt)
        else:
            print('There are no images in the table.')
    else:
        print('Two tables are not connected by any FOREIGN KEY')


save_employee_data('face_recognition/images/mysql_database')
