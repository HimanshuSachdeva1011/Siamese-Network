from keras.layers import Layer
import tensorflow as tf
from keras.models import load_model
import numpy as np
import os
import cv2

# GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
# Ignores those Big GPU Messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def preprocess(file_path):

    byte_img = tf.io.read_file(file_path)
    # read image
    img = tf.io.decode_jpeg(byte_img)
    # resize image to 105*105*3
    img = tf.image.resize(img, (105, 105))
    img = img/255.0

    return img


class L1Dist(Layer):
    def __int__(self, **kwargs):
        super(L1Dist, self).__int__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


model = load_model('siamese_model.h5', custom_objects={'L1Dist': L1Dist})


def verify(mdl, detection_threshold, verification_threshold):
    # Build results array
    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data', 'verification_images', image))

        # Make Predictions
        result = mdl.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    # Detection Threshold: Metric above which a prediction is considered positive
    detection = np.sum(np.array(results) > detection_threshold)

    # Verification Threshold: Proportion of positive predictions / total positive samples
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
    verified = verification > verification_threshold

    return results, verified


cam = cv2.VideoCapture(0)
while cam.isOpened():
    ret, frame = cam.read()
    frame = frame[120:120 + 250, 200:200 + 250, :]

    cv2.imshow('Verification', frame)

    # Verification trigger
    if cv2.waitKey(10) & 0xFF == ord('v'):
        # Save input image to application_data/input_image folder
        cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
        # Run verification
        results, verified = verify(model, 0.65, 0.65)
        print(verified)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

print(np.sum(np.squeeze(results) > 0.7))
