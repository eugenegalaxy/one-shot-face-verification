import cv2
import os
import re
import pyrealsense2 as rs
import numpy as np

CAM_IMG_DIR = 'new_entries'
CAM_IMG_DIR_MAX_SIZE_MB = 10  # In megabytes!!


def getWebcamImage(save=None, plot=None):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    ret, frame = cap.read()
    r = 250.0 / frame.shape[1]
    dim = (250, int(frame.shape[0] * r))
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    cap.release()

    if save is not None:
        path = CAM_IMG_DIR
        slash = '/'
        ext = '.jpg'
        name = 'image_'
        dir_size_guard()
        next_nr = generate_number_imgsave()
        full_name = path + slash + name + next_nr + ext
        cv2.imwrite(full_name, resized)
    if plot is not None:
        cv2.namedWindow('Webcam Photo', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Webcam Photo', resized)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return resized


def rs_video_feed_TEST():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)

    finally:

        # Stop streaming
        pipeline.stop()


def rs_init_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    for x in range(5):
        pipeline.wait_for_frames()
    return pipeline


def rs_capture_frame(pipeline):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    return color_image


def rs_getImage(save=None, plot=None):
    pipe = rs_init_camera()
    image = rs_capture_frame(pipe)
    pipe.stop()
    if save is not None:
        path = CAM_IMG_DIR
        slash = '/'
        dir_size_guard()
        ext = '.jpg'
        name = 'image_'
        next_nr = generate_number_imgsave()
        full_name = path + slash + name + next_nr + ext
        cv2.imwrite(full_name, image)
    if plot is not None:
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', image)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return image[..., ::-1]


def generate_number_imgsave(path=CAM_IMG_DIR):
    img_list = sorted(os.listdir(path))
    if len(img_list) == 0:
        return '0000'  # if folder is empty -> give next image '_0000' in name
    else:
        for word in list(img_list):  # iterating on a copy since removing will mess things up
            path_no_ext = os.path.splitext(str(word))[0]  # name without file extension like .jpg
            word_list = re.split('[/_.,:-]', str(path_no_ext))  # split name by char / or _ etc...
            if len(word_list) != 2 or word_list[0] != 'image':
                img_list.remove(word)

        img_last = img_list[-1]  # -1 -> last item in list
        path_no_ext = os.path.splitext(str(img_last))[0]  # name without file extension like .jpg
        word_list = re.split('[/_.,:-]', str(path_no_ext))
        next_number = int(word_list[1]) + 1
        next_number_str = str(next_number).zfill(4)
        return next_number_str


def dir_size_guard(path=CAM_IMG_DIR, limit_in_megabytes=CAM_IMG_DIR_MAX_SIZE_MB):
    while (dir_get_size() / 1048576) > limit_in_megabytes:
        file_list = sorted(os.listdir(path))
        if len(file_list) == 0:
            break
        print('Directory size reached limit of {0} megabytes. Deleting file "{1}".'.format(limit_in_megabytes, file_list[0]))
        os.remove(os.path.join(path, file_list[0]))


def dir_get_size(path=CAM_IMG_DIR):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size


#  CAREFUL!!! REMOVES ALL FILES IN A DIRECTORY
def dir_clear(path=CAM_IMG_DIR):
    file_list = sorted(os.listdir(path))
    # file_list.remove(file_list[0])  # Keep one file to avoid issues with uploading empty dir to Git.
    for file in file_list:
        os.remove(os.path.join(path, file))
