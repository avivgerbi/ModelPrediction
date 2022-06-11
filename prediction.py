import sys
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model

# Video preprocess
IMG_SIZE = 84
CHANNELS = 3
CHANNELS_OPTICAL_FLOW = 2
MAX_SEQ_LENGTH = 24


def load_video(path, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = (int(length / MAX_SEQ_LENGTH))
    counter = 0
    frames = []

    try:
        while True:
            counter += 1
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            if (counter % step == 0):
                frames.append(frame)
            if len(frames) == MAX_SEQ_LENGTH:
                break

    finally:
        cap.release()
    return np.array(frames)


def get_optical_flow(video_frames):
    gray_frames = []
    flows = []

    for i in range(len(video_frames)):
        img_float32 = np.float32(video_frames[i])
        gray_frame = cv2.cvtColor(img_float32, cv2.COLOR_RGB2GRAY)
        gray_frames.append(np.reshape(gray_frame, (IMG_SIZE, IMG_SIZE, 1)))

    for i in range(0, len(gray_frames) - 1):
        # calculate optical flow between each pair of frames
        flow = cv2.calcOpticalFlowFarneback(gray_frames[i], gray_frames[i + 1], None, 0.5, 3, 15, 3, 5, 1.2,
                                            cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        flow[..., 0] = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        flow[..., 1] = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)

        flows.append(flow)

    return np.array(flows)


def single_prediction(video):
    rgb_video = video
    optical_flow_video = get_optical_flow(video)

    rgb_video = tf.reshape(rgb_video, (1, MAX_SEQ_LENGTH, IMG_SIZE, IMG_SIZE, CHANNELS))
    optical_flow_video = tf.reshape(optical_flow_video,
                                    (1, MAX_SEQ_LENGTH - 1, IMG_SIZE, IMG_SIZE, CHANNELS_OPTICAL_FLOW))

    rgb_model_predication = load_model('./Models/rgb_model_new.h5')
    optical_flow_model_predication = load_model('./Models/optical_flow_model_new.h5')

    rgb_result = float(rgb_model_predication.predict(rgb_video))
    optical_flow_result = float(optical_flow_model_predication.predict(optical_flow_video))
    combine_result = rgb_result * 0.8 + optical_flow_result * 0.2

    return combine_result


print(single_prediction(load_video(str(sys.argv[1]))))
