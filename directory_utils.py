import numpy as np
import os.path
from lang_abbr_iso639_1 import languages
import re


class IdentityMetadata():
    def __init__(self, base, name, file):

        self.base = base  # dataset base directory
        self.name = name  # identity name
        self.file = file  # image file name

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file)


class IdentityMetadata_short():
    def __init__(self, base, file):

        self.base = base  # dataset base directory
        self.file = file  # image file name

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.file)


def load_metadata(path, names=None):
    metadata = []
    counter = 0  # Target counter
    for folder_name in sorted(os.listdir(path)):
        counter += 1
        for file_name in sorted(os.listdir(os.path.join(path, folder_name))):
            ext = os.path.splitext(file_name)[1]

            if names is not None:
                full_name_str = str()
                word_list = re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?\s]', str(folder_name))  # split words by special chars
                for word in word_list:
                    if word.count("") == 3 and [item for item in languages if item[0] == word]:
                        lang_full = [item[1] for item in languages if item[0] == word]
                        lang_main = lang_full[0].split(';')
                    else:
                        full_name_str += (" " + word)
                print(('Target {}:'.format(counter) + '\t').expandtabs(4) +
                      ('{}'.format(full_name_str) + '\t' + ' Language: {}'.format(lang_main[0])).expandtabs(30))
                lang_main = ['Unknown']
            if ext == '.jpg' or ext == '.jpeg':
                metadata.append(IdentityMetadata(path, folder_name, file_name))
    return np.array(metadata)


def load_metadata_short(path, names=None):
    metadata = []
    counter = 0  # Target counter
    for file_name in sorted(os.listdir(path)):
        counter += 1
        ext = os.path.splitext(file_name)[1]
        if ext == '.jpg' or ext == '.jpeg':
            metadata.append(IdentityMetadata_short(path, file_name))
    return np.array(metadata)


def find_language_code(single_metadata, print_text=None):
    language_found = False
    path_no_ext = os.path.splitext(str(single_metadata))[0]  # path without file extension like .jpg
    word_list = re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?\s]', str(path_no_ext))
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
        lang_full = 'English'
        if print_text is not None:
            print("Language not found. Set to English by default.")
    return lang, lang_full


def generate_number_imgsave(path):
    img_list = sorted(os.listdir(path))
    if len(img_list) == 0:
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
    bytes_in_one_megabyte = 1048576
    while (dir_get_size(path) / bytes_in_one_megabyte) > limit_in_megabytes:
        file_list = sorted(os.listdir(path))
        if len(file_list) == 0:
            break
        print('Directory size reached limit of {0} megabytes. Deleting file "{1}".'.format(limit_in_megabytes, file_list[0]))
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
    file_list = sorted(os.listdir(path))
    if save_one_file is not None:
        for file in file_list[1:]:
            os.remove(os.path.join(path, file))
    else:
        for file in file_list:
            os.remove(os.path.join(path, file))