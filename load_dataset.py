import numpy as np
import os.path
from lang_abbr_iso639_1 import languages
import re


class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file)


def load_metadata(path, names=None):
    metadata = []
    x = 0  # Target counter
    for i in sorted(os.listdir(path)):
        x += 1
        for f in sorted(os.listdir(os.path.join(path, i))):
            ext = os.path.splitext(f)[1]
            if names is not None:
                name = os.path.splitext(f)[0]
                print("Target {0}: {1}".format(x, name))
            if ext == '.jpg' or ext == '.jpeg':
                metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)

# def add_more_metadata(metadata, new_data_path, names=None)


def find_language_code(single_metadata, print_text=None):
    language_found = False
    path_no_ext = os.path.splitext(str(single_metadata))[0]  # path without file extension like .jpg
    word_list = re.split('[/_.,:-]', str(path_no_ext))
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
