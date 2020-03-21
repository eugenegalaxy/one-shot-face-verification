import cv2
import numpy as np
import matplotlib.pyplot as plt  # Plotting tool
from align import AlignDlib  # Face alignment method
from model import create_model  # CNN library


def align_image(img, alig_model):
    imgDim = 96
    face_bounding_box = alig_model.getLargestFaceBoundingBox(img)
    if face_bounding_box is None:
        return None
    face_keypoints = AlignDlib.OUTER_EYES_AND_NOSE
    aligned_img = alig_model.align(imgDim, img, face_bounding_box, landmarkIndices=face_keypoints)
    return aligned_img


def plot_opencv(img):
    cv2.namedWindow('Webcam Photo', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Webcam Photo', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def plot_plt(img):
    img = img[..., ::-1]
    plt.figure(num="Test")
    plt.imshow(img)
    plt.show()


def load_image(path):
    img = cv2.imread(path, 1)
    return img[..., ::-1]  # Reversing from BGR to RGB



# landmarks_pt = 'models/landmarks.dat'
# weights_pt = 'weights/nn4.small2.v1.h5'
# img_path = 'new_entries/image_0001.jpg'
# img_a = cv2.imread(img_path, 1)
# # img_b = load_image(img_path)
# # # plot_opencv(img_b)
# # # plot_plt(img_b)
# nn4_small2_pretrained = create_model()  # Create a Neural network model
# nn4_small2_pretrained.load_weights(weights_pt)  # Use pre-trained weights
# alig_model = AlignDlib(landmarks_pt)  # Initialize the OpenFace face alignment utility
# alig_model = AlignDlib(landmarks_pt)  # Initialize the OpenFace face alignment utility
# aimg = align_image(img_a, alig_model)
# if aimg is not None:
#     aimg = (aimg / 255.).astype(np.float32)  # scale RGB values to interval [0,1]
#     # plot_plt(aimg)
# else:
#     quit()
# embedded = np.zeros(3,128)
# emb_zero = np.full((1, 128), -0.1)
# embedded[2] = emb_zero

# embedded = nn4_small2_pretrained.predict(np.expand_dims(aimg, axis=0))[0]
# # print(embedded)


# def distance(feature1, feature2):
#     return np.sum(np.square(feature1 - feature2))





# print(distance(embedded, emb_zero))  # Sum of squared errors


embedded = np.zeros((3,128))
emb_zero = np.full((1, 128), -0.1)
embedded[2] = emb_zero
print(embedded)