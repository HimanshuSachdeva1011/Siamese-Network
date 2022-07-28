from keras import layers
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from keras.models import load_model
from keras.utils import Progbar
import tensorflow as tf
import cv2
import os
import numpy as np

# GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
# Ignores those Big GPU Messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# define anchor and positive path
anchor_path = os.path.join('data', 'anchor')
positive_path = os.path.join('data', 'positive')
negative_path = os.path.join('data', 'negative')


# Siamese L1 Distance class
class L1Dist(Layer):

    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


mdl = load_model('siamese_model.h5', custom_objects={'L1Dist': L1Dist})


# train test partition
def preprocess_twin(input_img, validation_img, label):
    return preprocess(input_img), preprocess(validation_img), label


def preprocess(file_path):

    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    # Resizing image to 105x105x3
    img = tf.image.resize(img, (105, 105))
    # Scaling image b/w 0 and 1
    img = img / 255.0

    return img


def verify(model, detection_threshold, verification_threshold):
    # Build results array
    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data', 'verification_images', image))

        # Make Predictions
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    # Detection Threshold: Metric above which a prediction is considered positive
    detection = np.sum(np.array(results) > detection_threshold)

    # Verification Threshold: Proportion of positive predictions / total positive samples
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
    verified = verification > verification_threshold

    return results, verified


cam = cv2.VideoCapture()
while cam.isOpened():
    ret, frame = cam.read()
    frame = frame[120:120+250:]
    cv2.imshow("verification", frame)
    if cv2.waitKey(1) & 0xFF:
        cv2.imwrite(os.path.join("application_data", "input_image", "input_image.jpg"))
