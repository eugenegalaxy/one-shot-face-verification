import numpy as np
import os.path


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


class IdentityMetadata_short():
    def __init__(self, base, file):
        # dataset base directory
        self.base = base
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.file)


def load_metadata_short(path, names=None):
    metadata = []
    x = 0  # Target counter
    for i in sorted(os.listdir(path)):
        x += 1
        ext = os.path.splitext(i)[1]
        if names is not None:
            name = os.path.splitext(i)[0]
            print("Target {0}: {1}".format(x, name))
        if ext == '.jpg' or ext == '.jpeg':
            metadata.append(IdentityMetadata_short(path, i))
    return np.array(metadata)


database = load_metadata('test_images', names=1)

print(database[1].name)

distances = [7,53,3,5,8,19,11,12,1,4,2]
smallest_dist = np.amin(distances)
smallest_idx = np.where(distances == np.amin(distances))
print("smallest_idx ", smallest_idx[0][0])
print(database)
most_similar_name = database[smallest_idx[0][0]].name
print("Target is most similar to: ", most_similar_name)