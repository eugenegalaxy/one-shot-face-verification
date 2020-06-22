#!/usr/bin/env python3
import os.path
import re
import numpy as np
import cv2

g_DEBUG_MODE = False  # Debug mode. Enables prints.

class IdentityMetadata():
    """Class to store metadata for dataset images

       Note: File must have following relative path -> "base/name/file"

       Example path: "test_dataset/Hugo_Markoff/img_0001.jpg"
    """
    def __init__(self, base, name, file):

        self.base = base  # dataset base directory
        self.name = name  # identity name
        self.file = file  # image file name

    def __repr__(self):
        return self.img_path()

    def img_path(self):
        return os.path.join(self.base, self.name, self.file)


class IdentityMetadata_short():
    """Class to store metadata for dataset images (SHORT VERSION)

       Note: File must have following relative path -> "base/file"

       Example path: "test_dataset/img_0001.jpg"
    """
    def __init__(self, base, file):

        self.base = base  # dataset base directory
        self.file = file  # image file name

    def __repr__(self):
        return self.img_path()

    def img_path(self):
        return os.path.join(self.base, self.file)


def load_metadata(path, names=None):
    """Load metadata into "IdentityMetadata" class objects.

    If arg "names" is provided, print all parsed labels and some information.

    Args:
        path (string): Path to a dataset of images
        names (anything but None, optional): Trigger to activate parsed name printing. Defaults to None.

    Returns:
        np.array: Numpy array with "IdentityMetadata" objects.
    """
    metadata = []
    counter = 0  # Target counter
    print('=====================================================================')
    print('Initializing directory: "{}"'.format(path))
    for folder_name in sorted(os.listdir(path)):
        for file_name in sorted(os.listdir(os.path.join(path, folder_name))):
            ext = os.path.splitext(file_name)[1]

            if names is not None:
                if ext == '.jpg' or ext == '.jpeg':
                    full_name_str = str()
                    lang_full = None

                    word_list = re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?\s]', str(folder_name))
                    for word in word_list:
                        if word.count("") == 3 and [item for item in languages if item[0] == word]:
                            lang_full = [item[1] for item in languages if item[0] == word]
                            lang_main = lang_full[0].split(';')
                        else:
                            full_name_str += (" " + word)

                    if lang_full is not None:
                        p1 = ('Image {}:'.format(counter) + '\t').expandtabs(4)
                        p2 = ('{}'.format(full_name_str) + '\t' + ' Language: {}'.format(lang_main[0])).expandtabs(30)
                        print(p1 + p2)
                    else:
                        print_nolang = ('Image {}:'.format(counter) + '\t').expandtabs(4) + ('{}'.format(full_name_str))
                        print(print_nolang)
                    lang_main = ['Unknown']

            if ext == '.jpg' or ext == '.jpeg':
                metadata.append(IdentityMetadata(path, folder_name, file_name))
                counter += 1
    print('=====================================================================')
    return np.array(metadata)


def load_metadata_short(path):
    """Load metadata into "IdentityMetadata" class objects.

    Args:
        path (string): Path to a dataset of images

    Returns:
        np.array: Numpy array with "IdentityMetadata" objects.
    """
    metadata = []
    counter = 0  # Target counter
    for file_name in sorted(os.listdir(path)):
        counter += 1
        ext = os.path.splitext(file_name)[1]
        if ext == '.jpg' or ext == '.jpeg':
            metadata.append(IdentityMetadata_short(path, file_name))
    return np.array(metadata)


def retrieve_info(path):
    """Looks for a text file in a provided path directory and reads lines into a dictionary.\

    WARNING: This function is designed for a specific, custom purpose: Read-in information on recognised victims.

    Args:
        path (string): Path to a labelled directory of some identity. Example: "test_dataset/Hugo_Markoff"

    Returns:
        text_data_dic (dictionary): A dictionary containing information about the recognised victim.
        image_path_list (list): A list containing paths of all images of the recognised victim.
    """
    if g_DEBUG_MODE is True:
        print('path: {}'.format(path))
    assert(os.path.isdir(path) is True), 'Provided path is not a directory!'

    image_path_list = []
    text_data = []
    text_data_dic = {}  # NOTE This is made for a specific case when text lines elements are in "type:data" format.
    person_name = []

    dir_name = os.path.basename(os.path.normpath(path))
    dir_name_split = re.split(r'[.`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,/<>?\s]', str(dir_name))
    for word in dir_name_split:
        if word.count("") == 3 and [item for item in languages if item[0] == word]:
            text_data_dic['languageCode'] = word
            nationality = [item[1] for item in languages if item[0] == word]
            if nationality[0].find(';') != -1:
                nationality_special_char_idx = nationality[0].find(';')
                text_data_dic['nationality'] = nationality[0][:nationality_special_char_idx]
            else:
                text_data_dic['nationality'] = nationality[0]
            if text_data_dic['nationality'] in locale:
                text_data_dic['voiceRec'] = locale[text_data_dic['nationality']]
            else:
                text_data_dic['voiceRec'] = 'en-US'
        else:
            person_name.append(word)
            text_data_dic['nationality'] = 'English'
            text_data_dic['languageCode'] = 'en'
            text_data_dic['voiceRec'] = 'en-US'
    person_name = ' '.join(person_name)
    text_data_dic['fullName'] = person_name

    for file_name in sorted(os.listdir(path)):

        ext = os.path.splitext(file_name)[1]
        if ext == '.jpg' or ext == '.jpeg':
            image_path = os.path.join(path, file_name)
            image_path_list.append(image_path)
        # elif ext == '.txt': # This one doesnt work together with new feature .txt files.
        elif file_name == 'info.txt':
            text_path = os.path.join(path, file_name)
            with open(text_path) as f:
                lines = [line.rstrip('\n') for line in f]
                text_data.append(lines)
        elif os.path.isdir(os.path.join(path, file_name)) is True:
            if g_DEBUG_MODE is True:
                print('Information Retrieval -> Directory "{}" found. Ignoring it.'.format(file_name))
        else:
            if g_DEBUG_MODE is True:
                print('Information Retrieval -> File "{}" found. Ignoring it.'.format(file_name))
    text_data = [item for sublist in text_data for item in sublist]  # Flatten list of lists into a list.
    for item in text_data:
        ignore_strings = ['www', 'http']
        if any(x in item for x in ignore_strings):   # Special treatment for list element containing website link.
            semicolon_index = item.find(':')         # Example:    'abc:123' -> index is 4
            first_str = item[:semicolon_index]       # First Str:  'abc'
            second_str = item[semicolon_index + 1:]  # Second Str: '123'
            text_data_dic[first_str] = second_str
        else:
            # NOTE Separator symbol is inside r'[]' -> used only : to avoid stripping website link in lists.
            word_list = re.split(r'[:]', item)
            if word_list[0] == 'nationality':
                nationality = word_list[1].strip()
                if nationality in locale:
                    text_data_dic['voiceRec'] = locale[nationality]
                else:
                    text_data_dic['voiceRec'] = 'en-US'
            text_data_dic[word_list[0]] = word_list[1]

    return text_data_dic, image_path_list


def resize_img(img, dim=None, scale=None, adjust_to_width=None):
    if dim is not None:
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    elif scale is not None:
        scale_percent = scale  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    elif adjust_to_width is not None:
        res = adjust_to_width / img.shape[1]
        dim = (adjust_to_width, int(img.shape[0] * res))
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    else:
        raise ValueError('No resizing mode is selected.')
    return resized

def resize_all_in_folder(path, dim=None, scale=None, adjust_to_width=None, save=None, plot=None):
    resized_imgs = []
    dirs = os.listdir(path)
    for item in dirs:
        f, ext = os.path.splitext(path + item)
        if ext == '.jpg' or ext == '.jpeg':
            img = cv2.imread(path + item, 1)
            if dim is not None:
                resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            elif scale is not None:
                scale_percent = scale  # percent of original size
                width = int(img.shape[1] * scale_percent / 100)
                height = int(img.shape[0] * scale_percent / 100)
                dim = (width, height)
                resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            elif adjust_to_width is not None:
                res = adjust_to_width / img.shape[1]
                dim = (adjust_to_width, int(img.shape[0] * res))
                resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            else:
                raise ValueError('No resizing mode is selected.')

            if save is not None:
                full_path = f + '_resized' + ext
                cv2.imwrite(full_path, resized)

            if plot is not None:
                cv2.imshow("Resized image {0}x{1}".format(dim[0], dim[1]), resized)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            resized_imgs.append(resized)
    return resized_imgs


def find_language_code(single_metadata, print_text=None):
    """Checks for alpha-2 code letters in directory or file name and assigns a language.

    Example: "Eugene_lv/image_0001.jpg" will return 'lv, "Latvian" '.

    Args:
        single_metadata (string): Path to a labelled directory of an identity.
        print_text (anything but None, optional): Trigger to activate printing. Defaults to None.

    Returns:
        lang (string): alpha-2 language code
        lang_full (string): Full language name
    """
    language_found = False
    path_no_ext = os.path.splitext(str(single_metadata))[0]  # path without file extension like .jpg
    word_list = re.split(r'[.`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,/<>?\s]', str(path_no_ext))
    for word in word_list:
        if word.count("") == 3 and [item for item in languages if item[0] == word]:
            lang_full = [item[1] for item in languages if item[0] == word]
            lang = word
            language_found = True
            break  # If language code is found in folder name, no need to search further in file name
            if print_text is not None:
                print("Language code found '{0}': {1}.".format(lang, lang_full[0]))
    if language_found is False:
        lang = 'en'
        lang_full = ['English']
        if print_text is not None:
            print("Language not found. Set to English by default.")
    return lang, lang_full


def generate_number_imgsave(path):
    """Consecutive number generator for image names. Scans provided directory and determines largest number.

    If no images in the provided directory, will start from "0000"


    Args:
        path (string): Path to a directory

    Returns:
        string: Four digit number from 0000 to 9999
    """
    img_list = sorted(os.listdir(path))
    img_number = 0
    for item in img_list:
        ext = os.path.splitext(item)[1]
        if ext == '.jpg' or ext == '.jpeg':
            img_number += 1
    if img_number == 0:
        return '0000'  # if folder is empty -> give next image '_0000' in name
    else:
        for word in list(img_list):  # iterating on a copy since removing will mess things up
            path_no_ext = os.path.splitext(str(word))[0]  # name without file extension like .jpg
            word_list = re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?\s]', str(path_no_ext))
            if len(word_list) != 2 or word_list[0] != 'image':
                img_list.remove(word)

        img_last = img_list[-1]  # -1 -> last item in list
        path_no_ext = os.path.splitext(str(img_last))[0]
        word_list = re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?\s]', str(path_no_ext))
        next_number = int(word_list[1]) + 1
        next_number_str = str(next_number).zfill(4)
        return next_number_str


def dir_size_guard(path, limit_in_megabytes):
    """ Check directory size against provided limit and deletes first files in alphabetical order.

    Note: Assuming all images have autogenerated name, e.g. "image_0001.jpg", the most earlier images
    will be deleted.

    Args:
        path (string): Path to a directory
        limit_in_megabytes (int): Maximum number of megabytes for the provided "path" directory.
    """
    bytes_in_one_megabyte = 1048576
    while (dir_get_size(path) / bytes_in_one_megabyte) > limit_in_megabytes:
        file_list = sorted(os.listdir(path))
        if len(file_list) == 0:
            break
        print('Directory size reached limit of {0} megabytes. Deleting file "{1}".'.format(
            limit_in_megabytes, file_list[0]))
        os.remove(os.path.join(path, file_list[0]))


def dir_get_size(path):

    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size


#  CAREFUL!!! REMOVES ALL FILES IN A DIRECTORY
def dir_clear(path, save_one_file=None):
    """DOOM function to remove all files in a folder. Just a convinience tool.

    Args:
        path (string): Path to a directory
        save_one_file (anything but None, optional): Keep one file alive so git will keep track of the directory.
    """
    file_list = sorted(os.listdir(path))
    if save_one_file is not None:
        for file in file_list[1:]:
            os.remove(os.path.join(path, file))
    else:
        for file in file_list:
            os.remove(os.path.join(path, file))


def trim_list_std(list_input, lower_std, upper_std):
    """ Removes values in a list that are below "lower_std" and above "upper_std".

    Args:
        list_input (list): List of numbers
        lower_std (float): Negative standard deviation coefficient.
        upper_std (float): Positive standard deviation coefficient.

    Returns:
        list: Filtered list of numbers.
    """
    list_input = np.array(list_input)
    mean = np.mean(list_input)
    sd = np.std(list_input)
    final_list = [x for x in list_input if (x > mean - lower_std * sd)]
    final_list = [x for x in final_list if (x < mean + upper_std * sd)]
    return final_list

# coding: utf8
# ISO639-3 language codesin (alpha_2,language_name) format.
languages = [
    ('aa', 'Afar'),
    ('ab', 'Abkhazian'),
    ('af', 'Afrikaans'),
    ('ak', 'Akan'),
    ('sq', 'Albanian'),
    ('am', 'Amharic'),
    ('ar', 'Arabic'),
    ('an', 'Aragonese'),
    ('hy', 'Armenian'),
    ('as', 'Assamese'),
    ('av', 'Avaric'),
    ('ae', 'Avestan'),
    ('ay', 'Aymara'),
    ('az', 'Azerbaijani'),
    ('ba', 'Bashkir'),
    ('bm', 'Bambara'),
    ('eu', 'Basque'),
    ('be', 'Belarusian'),
    ('bn', 'Bengali'),
    ('bh', 'Bihari languages'),
    ('bi', 'Bislama'),
    ('bo', 'Tibetan'),
    ('bs', 'Bosnian'),
    ('br', 'Breton'),
    ('bg', 'Bulgarian'),
    ('my', 'Burmese'),
    ('ca', 'Catalan; Valencian'),
    ('cs', 'Czech'),
    ('ch', 'Chamorro'),
    ('ce', 'Chechen'),
    ('zh', 'Chinese'),
    ('cu', 'Church Slavic; Old Slavonic; Church Slavonic; Old Bulgarian; Old Church Slavonic'),
    ('cv', 'Chuvash'),
    ('kw', 'Cornish'),
    ('co', 'Corsican'),
    ('cr', 'Cree'),
    ('cy', 'Welsh'),
    ('cs', 'Czech'),
    ('da', 'Danish'),
    ('de', 'German'),
    ('dv', 'Divehi; Dhivehi; Maldivian'),
    ('nl', 'Dutch; Flemish'),
    ('dz', 'Dzongkha'),
    ('el', 'Greek, Modern (1453-)'),
    ('en', 'English'),
    ('eo', 'Esperanto'),
    ('et', 'Estonian'),
    ('eu', 'Basque'),
    ('ee', 'Ewe'),
    ('fo', 'Faroese'),
    ('fa', 'Persian'),
    ('fj', 'Fijian'),
    ('fi', 'Finnish'),
    ('fr', 'French'),
    ('fy', 'Western Frisian'),
    ('ff', 'Fulah'),
    ('Ga', 'Georgian'),
    ('de', 'German'),
    ('gd', 'Gaelic; Scottish Gaelic'),
    ('ga', 'Irish'),
    ('gl', 'Galician'),
    ('gv', 'Manx'),
    ('el', 'Greek, Modern (1453-)'),
    ('gn', 'Guarani'),
    ('gu', 'Gujarati'),
    ('ht', 'Haitian; Haitian Creole'),
    ('ha', 'Hausa'),
    ('he', 'Hebrew'),
    ('hz', 'Herero'),
    ('hi', 'Hindi'),
    ('ho', 'Hiri Motu'),
    ('hr', 'Croatian'),
    ('hu', 'Hungarian'),
    ('hy', 'Armenian'),
    ('ig', 'Igbo'),
    ('is', 'Icelandic'),
    ('io', 'Ido'),
    ('ii', 'Sichuan Yi; Nuosu'),
    ('iu', 'Inuktitut'),
    ('ie', 'Interlingue; Occidental'),
    ('ia', 'Interlingua (International Auxiliary Language Association)'),
    ('id', 'Indonesian'),
    ('ik', 'Inupiaq'),
    ('is', 'Icelandic'),
    ('it', 'Italian'),
    ('jv', 'Javanese'),
    ('ja', 'Japanese'),
    ('kl', 'Kalaallisut; Greenlandic'),
    ('kn', 'Kannada'),
    ('ks', 'Kashmiri'),
    ('ka', 'Georgian'),
    ('kr', 'Kanuri'),
    ('kk', 'Kazakh'),
    ('km', 'Central Khmer'),
    ('ki', 'Kikuyu; Gikuyu'),
    ('rw', 'Kinyarwanda'),
    ('ky', 'Kirghiz; Kyrgyz'),
    ('kv', 'Komi'),
    ('kg', 'Kongo'),
    ('ko', 'Korean'),
    ('kj', 'Kuanyama; Kwanyama'),
    ('ku', 'Kurdish'),
    ('lo', 'Lao'),
    ('la', 'Latin'),
    ('lv', 'Latvian'),
    ('li', 'Limburgan; Limburger; Limburgish'),
    ('ln', 'Lingala'),
    ('lt', 'Lithuanian'),
    ('lb', 'Luxembourgish; Letzeburgesch'),
    ('lu', 'Luba-Katanga'),
    ('lg', 'Ganda'),
    ('mk', 'Macedonian'),
    ('mh', 'Marshallese'),
    ('ml', 'Malayalam'),
    ('mi', 'Maori'),
    ('mr', 'Marathi'),
    ('ms', 'Malay'),
    ('Mi', 'Micmac'),
    ('mk', 'Macedonian'),
    ('mg', 'Malagasy'),
    ('mt', 'Maltese'),
    ('mn', 'Mongolian'),
    ('mi', 'Maori'),
    ('ms', 'Malay'),
    ('my', 'Burmese'),
    ('na', 'Nauru'),
    ('nv', 'Navajo; Navaho'),
    ('nr', 'Ndebele, South; South Ndebele'),
    ('nd', 'Ndebele, North; North Ndebele'),
    ('ng', 'Ndonga'),
    ('ne', 'Nepali'),
    ('nl', 'Dutch; Flemish'),
    ('nn', 'Norwegian Nynorsk; Nynorsk, Norwegian'),
    ('nb', 'Bokmål, Norwegian; Norwegian Bokmål'),
    ('no', 'Norwegian'),
    ('oc', 'Occitan (post 1500)'),
    ('oj', 'Ojibwa'),
    ('or', 'Oriya'),
    ('om', 'Oromo'),
    ('os', 'Ossetian; Ossetic'),
    ('pa', 'Panjabi; Punjabi'),
    ('fa', 'Persian'),
    ('pi', 'Pali'),
    ('pl', 'Polish'),
    ('pt', 'Portuguese'),
    ('ps', 'Pushto; Pashto'),
    ('qu', 'Quechua'),
    ('rm', 'Romansh'),
    ('ro', 'Romanian; Moldavian; Moldovan'),
    ('rn', 'Rundi'),
    ('ru', 'Russian'),
    ('sg', 'Sango'),
    ('sa', 'Sanskrit'),
    ('si', 'Sinhala; Sinhalese'),
    ('sk', 'Slovak'),
    ('sk', 'Slovak'),
    ('sl', 'Slovenian'),
    ('se', 'Northern Sami'),
    ('sm', 'Samoan'),
    ('sn', 'Shona'),
    ('sd', 'Sindhi'),
    ('so', 'Somali'),
    ('st', 'Sotho, Southern'),
    ('es', 'Spanish; Castilian'),
    ('sq', 'Albanian'),
    ('sc', 'Sardinian'),
    ('sr', 'Serbian'),
    ('ss', 'Swati'),
    ('su', 'Sundanese'),
    ('sw', 'Swahili'),
    ('sv', 'Swedish'),
    ('ty', 'Tahitian'),
    ('ta', 'Tamil'),
    ('tt', 'Tatar'),
    ('te', 'Telugu'),
    ('tg', 'Tajik'),
    ('tl', 'Tagalog'),
    ('th', 'Thai'),
    ('bo', 'Tibetan'),
    ('ti', 'Tigrinya'),
    ('to', 'Tonga (Tonga Islands)'),
    ('tn', 'Tswana'),
    ('ts', 'Tsonga'),
    ('tk', 'Turkmen'),
    ('tr', 'Turkish'),
    ('tw', 'Twi'),
    ('ug', 'Uighur; Uyghur'),
    ('uk', 'Ukrainian'),
    ('ur', 'Urdu'),
    ('uz', 'Uzbek'),
    ('ve', 'Venda'),
    ('vi', 'Vietnamese'),
    ('vo', 'Volapük'),
    ('cy', 'Welsh'),
    ('wa', 'Walloon'),
    ('wo', 'Wolof'),
    ('xh', 'Xhosa'),
    ('yi', 'Yiddish'),
    ('yo', 'Yoruba'),
    ('za', 'Zhuang; Chuang'),
    ('zh', 'Chinese'),
    ('zu', 'Zulu')
]
# ISO 3166 country codes in (alpha_2,country_name) format.
countries = [
    ('AF', u'Afghanistan'),
    ('AX', u'\xc5land Islands'),
    ('AL', u'Albania'),
    ('DZ', u'Algeria'),
    ('AS', u'American Samoa'),
    ('AD', u'Andorra'),
    ('AO', u'Angola'),
    ('AI', u'Anguilla'),
    ('AQ', u'Antarctica'),
    ('AG', u'Antigua and Barbuda'),
    ('AR', u'Argentina'),
    ('AM', u'Armenia'),
    ('AW', u'Aruba'),
    ('AU', u'Australia'),
    ('AT', u'Austria'),
    ('AZ', u'Azerbaijan'),
    ('BS', u'Bahamas'),
    ('BH', u'Bahrain'),
    ('BD', u'Bangladesh'),
    ('BB', u'Barbados'),
    ('BY', u'Belarus'),
    ('BE', u'Belgium'),
    ('BZ', u'Belize'),
    ('BJ', u'Benin'),
    ('BM', u'Bermuda'),
    ('BT', u'Bhutan'),
    ('BO', u'Bolivia, Plurinational State of'),
    ('BQ', u'Bonaire, Sint Eustatius and Saba'),
    ('BA', u'Bosnia and Herzegovina'),
    ('BW', u'Botswana'),
    ('BV', u'Bouvet Island'),
    ('BR', u'Brazil'),
    ('IO', u'British Indian Ocean Territory'),
    ('BN', u'Brunei Darussalam'),
    ('BG', u'Bulgaria'),
    ('BF', u'Burkina Faso'),
    ('BI', u'Burundi'),
    ('KH', u'Cambodia'),
    ('CM', u'Cameroon'),
    ('CA', u'Canada'),
    ('CV', u'Cape Verde'),
    ('KY', u'Cayman Islands'),
    ('CF', u'Central African Republic'),
    ('TD', u'Chad'),
    ('CL', u'Chile'),
    ('CN', u'China'),
    ('CX', u'Christmas Island'),
    ('CC', u'Cocos (Keeling Islands)'),
    ('CO', u'Colombia'),
    ('KM', u'Comoros'),
    ('CG', u'Congo'),
    ('CD', u'Congo, The Democratic Republic of the'),
    ('CK', u'Cook Islands'),
    ('CR', u'Costa Rica'),
    ('CI', u"C\xf4te D'ivoire"),
    ('HR', u'Croatia'),
    ('CU', u'Cuba'),
    ('CW', u'Cura\xe7ao'),
    ('CY', u'Cyprus'),
    ('CZ', u'Czech Republic'),
    ('DK', u'Denmark'),
    ('DJ', u'Djibouti'),
    ('DM', u'Dominica'),
    ('DO', u'Dominican Republic'),
    ('EC', u'Ecuador'),
    ('EG', u'Egypt'),
    ('SV', u'El Salvador'),
    ('GQ', u'Equatorial Guinea'),
    ('ER', u'Eritrea'),
    ('EE', u'Estonia'),
    ('ET', u'Ethiopia'),
    ('FK', u'Falkland Islands (Malvinas)'),
    ('FO', u'Faroe Islands'),
    ('FJ', u'Fiji'),
    ('FI', u'Finland'),
    ('FR', u'France'),
    ('GF', u'French Guiana'),
    ('PF', u'French Polynesia'),
    ('TF', u'French Southern Territories'),
    ('GA', u'Gabon'),
    ('GM', u'Gambia'),
    ('GE', u'Georgia'),
    ('DE', u'Germany'),
    ('GH', u'Ghana'),
    ('GI', u'Gibraltar'),
    ('GR', u'Greece'),
    ('GL', u'Greenland'),
    ('GD', u'Grenada'),
    ('GP', u'Guadeloupe'),
    ('GU', u'Guam'),
    ('GT', u'Guatemala'),
    ('GG', u'Guernsey'),
    ('GN', u'Guinea'),
    ('GW', u'Guinea-bissau'),
    ('GY', u'Guyana'),
    ('HT', u'Haiti'),
    ('HM', u'Heard Island and McDonald Islands'),
    ('VA', u'Holy See (Vatican City State)'),
    ('HN', u'Honduras'),
    ('HK', u'Hong Kong'),
    ('HU', u'Hungary'),
    ('IS', u'Iceland'),
    ('IN', u'India'),
    ('ID', u'Indonesia'),
    ('IR', u'Iran, Islamic Republic of'),
    ('IQ', u'Iraq'),
    ('IE', u'Ireland'),
    ('IM', u'Isle of Man'),
    ('IL', u'Israel'),
    ('IT', u'Italy'),
    ('JM', u'Jamaica'),
    ('JP', u'Japan'),
    ('JE', u'Jersey'),
    ('JO', u'Jordan'),
    ('KZ', u'Kazakhstan'),
    ('KE', u'Kenya'),
    ('KI', u'Kiribati'),
    ('KP', u"Korea, Democratic People's Republic of"),
    ('KR', u'Korea, Republic of'),
    ('KW', u'Kuwait'),
    ('KG', u'Kyrgyzstan'),
    ('LA', u"Lao People's Democratic Republic"),
    ('LV', u'Latvia'),
    ('LB', u'Lebanon'),
    ('LS', u'Lesotho'),
    ('LR', u'Liberia'),
    ('LY', u'Libya'),
    ('LI', u'Liechtenstein'),
    ('LT', u'Lithuania'),
    ('LU', u'Luxembourg'),
    ('MO', u'Macao'),
    ('MK', u'Macedonia, The Former Yugoslav Republic of'),
    ('MG', u'Madagascar'),
    ('MW', u'Malawi'),
    ('MY', u'Malaysia'),
    ('MV', u'Maldives'),
    ('ML', u'Mali'),
    ('MT', u'Malta'),
    ('MH', u'Marshall Islands'),
    ('MQ', u'Martinique'),
    ('MR', u'Mauritania'),
    ('MU', u'Mauritius'),
    ('YT', u'Mayotte'),
    ('MX', u'Mexico'),
    ('FM', u'Micronesia, Federated States of'),
    ('MD', u'Moldova, Republic of'),
    ('MC', u'Monaco'),
    ('MN', u'Mongolia'),
    ('ME', u'Montenegro'),
    ('MS', u'Montserrat'),
    ('MA', u'Morocco'),
    ('MZ', u'Mozambique'),
    ('MM', u'Myanmar'),
    ('NA', u'Namibia'),
    ('NR', u'Nauru'),
    ('NP', u'Nepal'),
    ('NL', u'Netherlands'),
    ('NC', u'New Caledonia'),
    ('NZ', u'New Zealand'),
    ('NI', u'Nicaragua'),
    ('NE', u'Niger'),
    ('NG', u'Nigeria'),
    ('NU', u'Niue'),
    ('NF', u'Norfolk Island'),
    ('MP', u'Northern Mariana Islands'),
    ('NO', u'Norway'),
    ('OM', u'Oman'),
    ('PK', u'Pakistan'),
    ('PW', u'Palau'),
    ('PS', u'Palestinian Territory, Occupied'),
    ('PA', u'Panama'),
    ('PG', u'Papua New Guinea'),
    ('PY', u'Paraguay'),
    ('PE', u'Peru'),
    ('PH', u'Philippines'),
    ('PN', u'Pitcairn'),
    ('PL', u'Poland'),
    ('PT', u'Portugal'),
    ('PR', u'Puerto Rico'),
    ('QA', u'Qatar'),
    ('RE', u'R\xe9union'),
    ('RO', u'Romania'),
    ('RU', u'Russian Federation'),
    ('RW', u'Rwanda'),
    ('BL', u'Saint Barth\xe9lemy'),
    ('SH', u'Saint Helena, Ascension and Tristan Da Cunha'),
    ('KN', u'Saint Kitts and Nevis'),
    ('LC', u'Saint Lucia'),
    ('MF', u'Saint Martin (French Part)'),
    ('PM', u'Saint Pierre and Miquelon'),
    ('VC', u'Saint Vincent and the Grenadines'),
    ('WS', u'Samoa'),
    ('SM', u'San Marino'),
    ('ST', u'Sao Tome and Principe'),
    ('SA', u'Saudi Arabia'),
    ('SN', u'Senegal'),
    ('RS', u'Serbia'),
    ('SC', u'Seychelles'),
    ('SL', u'Sierra Leone'),
    ('SG', u'Singapore'),
    ('SX', u'Sint Maarten (Dutch Part)'),
    ('SK', u'Slovakia'),
    ('SI', u'Slovenia'),
    ('SB', u'Solomon Islands'),
    ('SO', u'Somalia'),
    ('ZA', u'South Africa'),
    ('GS', u'South Georgia and the South Sandwich Islands'),
    ('SS', u'South Sudan'),
    ('ES', u'Spain'),
    ('LK', u'Sri Lanka'),
    ('SD', u'Sudan'),
    ('SR', u'Suriname'),
    ('SJ', u'Svalbard and Jan Mayen'),
    ('SZ', u'Swaziland'),
    ('SE', u'Sweden'),
    ('CH', u'Switzerland'),
    ('SY', u'Syrian Arab Republic'),
    ('TW', u'Taiwan, Province of China'),
    ('TJ', u'Tajikistan'),
    ('TZ', u'Tanzania, United Republic of'),
    ('TH', u'Thailand'),
    ('TL', u'Timor-leste'),
    ('TG', u'Togo'),
    ('TK', u'Tokelau'),
    ('TO', u'Tonga'),
    ('TT', u'Trinidad and Tobago'),
    ('TN', u'Tunisia'),
    ('TR', u'Turkey'),
    ('TM', u'Turkmenistan'),
    ('TC', u'Turks and Caicos Islands'),
    ('TV', u'Tuvalu'),
    ('UG', u'Uganda'),
    ('UA', u'Ukraine'),
    ('AE', u'United Arab Emirates'),
    ('GB', u'United Kingdom'),
    ('US', u'United States'),
    ('UM', u'United States Minor Outlying Islands'),
    ('UY', u'Uruguay'),
    ('UZ', u'Uzbekistan'),
    ('VU', u'Vanuatu'),
    ('VE', u'Venezuela, Bolivarian Republic of'),
    ('VN', u'Viet Nam'),
    ('VG', u'Virgin Islands, British'),
    ('VI', u'Virgin Islands, U.S.'),
    ('WF', u'Wallis and Futuna'),
    ('EH', u'Western Sahara'),
    ('YE', u'Yemen'),
    ('ZM', u'Zambia'),
    ('ZW', u'Zimbabwe')
]


locale = {
    'Bulgarian': 'bg-BG',
    'Czech': 'cs-CZ',
    'Danish': 'da-DK',
    'German': 'de-DE',
    'Greek': 'el-GR',
    'English': 'en-US',
    'Spanish': 'es-ES',
    'Estonian': 'et-EE',
    'Finnish': 'fi-FI',
    'French': 'fr-FR',
    'Croatian': 'hr-HR',
    'Hungarian': 'hu-HU',
    'Italian': 'it-IT',
    'Lithuanian': 'lt-LT',
    'Latvian': 'lv-LV',
    'Dutch': 'nl-NL',
    'Norwegian': 'no-NO',
    'Polish': 'pl-PL',
    'Portuguese': 'pt-PT',
    'Romanian': 'ro-RO',
    'Russian': 'ru-RU',
    'Slovak': 'sk-SK',
    'Slovenian': 'sl-SI',
    'Swedish': 'sv-SE',
    'Turkish': 'tr-TR',
    'Chinese': 'zh-CN',
}
