#!/usr/bin/env python3
import cv2
import pyrealsense2 as rs
import numpy as np
from face_recognition.directory_utils import *
import time

CAM_IMG_DIR_MAX_SIZE_MB = 200  # In megabytes!!


def getImg_webcam(save_path=None, plot=None):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    ret, frame = cap.read()
    r = 250.0 / frame.shape[1]
    dim = (250, int(frame.shape[0] * r))
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    cap.release()

    if save_path is not None:
        path = save_path
        slash = '/'
        ext = '.jpg'
        name = 'image_'
        dir_size_guard(path, CAM_IMG_DIR_MAX_SIZE_MB)
        next_nr = generate_number_imgsave(path)
        full_name = path + slash + name + next_nr + ext
        cv2.imwrite(full_name, resized)
    if plot is not None:
        cv2.namedWindow('Webcam Photo', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Webcam Photo', resized)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return resized


def getImg_realsense(save_path=None, plot=None):

    # Initiliase camera, get pipe
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    # Throw away some frames to stabilize parameters (exposure, light, etc)
    for x in range(5):
        pipeline.wait_for_frames()

    time.sleep(1)

    # Capture frame
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    image = np.asanyarray(color_frame.get_data())

    pipeline.stop()

    if save_path is not None:
        path = save_path
        slash = '/'
        dir_size_guard(path, CAM_IMG_DIR_MAX_SIZE_MB)
        ext = '.jpg'
        name = 'image_'
        next_nr = generate_number_imgsave(path)
        full_name = path + slash + name + next_nr + ext
        cv2.imwrite(full_name, image)

    if plot is not None:
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', image)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return image


def getManyImg_webcam(numberImg, time_interval_sec, save_path):  # TODO: Not tested if works fine with IntelReal Sense
    img_counter = 0
    for x in range(numberImg):
        getImg_webcam(save_path=save_path)
        img_counter += 1
        remaining = numberImg - img_counter
        print('Photo {0} is captured. Remaining {1}. Saving in "{2}".'.format(img_counter, remaining, save_path))
        time.sleep(time_interval_sec)


def getManyImg_realsense(numberImg, time_interval_sec, save_path):  # TODO: Not tested if works fine with IntelReal Sense
    img_counter = 0
    for x in range(numberImg):
        getImg_realsense(save_path=save_path)
        img_counter += 1
        remaining = numberImg - img_counter
        print('Photo {0} is captured. Remaining {1}. Saving in "{2}".'.format(img_counter, remaining, save_path))
        time.sleep(time_interval_sec)
