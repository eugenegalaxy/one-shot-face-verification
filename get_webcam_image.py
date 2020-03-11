import cv2
import time


def get_webcam_image():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    ret, frame = cap.read()
    r = 250.0 / frame.shape[1]
    dim = (250, int(frame.shape[0] * r))
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite('new_entries/person/image.jpg', resized)
    cap.release()
    cv2.destroyAllWindows()
    return resized
