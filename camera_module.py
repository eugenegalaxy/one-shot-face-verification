import cv2
import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt  # Plotting tool
import time


def get_webcam_image(save=None, plot=None):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    ret, frame = cap.read()
    r = 250.0 / frame.shape[1]
    dim = (250, int(frame.shape[0] * r))
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    cap.release()
    if save is not None:
        cv2.imwrite('new_entries/person/image.jpg', resized)
    if plot is not None:
        cv2.namedWindow('Webcam Photo', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Webcam Photo', resized)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return resized[..., ::-1]


def video_feed_TEST():
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


def getImage(save=None, plot=None):
    pipe = rs_init_camera()
    image = rs_capture_frame(pipe)
    pipe.stop()
    if save is not None:
        cv2.imwrite('new_entries/person/image.jpg', image)
    if plot is not None:
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', image)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return image[..., ::-1]


def load_image(path):
    img = cv2.imread(path, 1)
    return img[..., ::-1]  # Reversing from BGR to RGB


