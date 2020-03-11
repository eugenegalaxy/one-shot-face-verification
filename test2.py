from load_dataset import load_metadata  # Loading training set in from folder
import cv2
import numpy as np
from align import AlignDlib  # Face alignment method
from model import create_model  # CNN library


def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[..., ::-1]


def align_image(img, alig_model):
    return alig_model.align(96, img, alig_model.getLargestFaceBoundingBox(img),
                            landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)


def run_CNN(metadata, face_align_model):
    embedded = np.zeros((metadata.shape[0], 128))
    for i, m in enumerate(metadata):
        img = load_image(m.image_path())
        img = align_image(img, face_align_model)

        # plt.imshow(img)
        # plt.title('Aligned')
        # plt.show()

        # scale RGB values to interval [0,1]
        img = (img / 255.).astype(np.float32)
        # obtain embedding vector for image
        embedded[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]
    return embedded


# Create a Neural network model (structure from model.py)
nn4_small2_pretrained = create_model()
# Use pre-trained weights
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')
# Initialize the OpenFace face alignment utility
alignment_model = AlignDlib('models/landmarks.dat')

database = load_metadata('test_images', names=1)
emb_database = run_CNN(database, alignment_model)

new = load_metadata('new_entries', names=1)
emb_new = run_CNN(new, alignment_model)

database = np.append(database, new)
emb_database = np.append(emb_database, emb_new, axis=0)
print(database.shape)
print(emb_database.shape)
