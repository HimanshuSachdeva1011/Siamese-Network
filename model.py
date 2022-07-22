import os

import keras.losses
from keras import losses, optimizers
from keras.models import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
from tensorflow import train

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# define anchor and positive path
anchor_path = os.path.join('data', 'anchor')
positive_path = os.path.join('data', 'positive')
negative_path = os.path.join('data', 'negative')

# Load the dataset
anchor = tf.data.Dataset.list_files(anchor_path+'\*.jpg').take(58)
positive = tf.data.Dataset.list_files(positive_path+'\*.jpg').take(58)
negative = tf.data.Dataset.list_files(negative_path+'\*.jpg').take(58)


def preprocess(file_path):

    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)

    # Resizing the image to be 105x105x3
    img = tf.image.resize(img, (105, 105))
    # Scaling the image between 0 and 1
    img = img / 255.0

    return img


# Label the dataset
# pos_ve = ones of anchor length
# neg_ve = zeros of anchor length
pos_ve = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
neg_ve = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))

# Concatenate files to data variable
data = pos_ve.concatenate(neg_ve)


# Train And Test Partition
def preprocess_twin(input_img, validation_img, label):
    return preprocess(input_img), preprocess(validation_img), label


# Dataloader Pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)

# Train Partition
train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

# Test partition
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)


def make_model_embedding():
    inp = Input(shape=(105, 105, 3), name='input_image')

    conv1 = Conv2D(64, (10, 10), activation='relu')(inp)
    max1 = MaxPooling2D(64, (2, 2), padding='same')(conv1)

    conv2 = Conv2D(128, (7, 7), activation='relu')(max1)
    m2 = MaxPooling2D(64, (2, 2), padding='same')(conv2)

    conv3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding='same')(conv3)

    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=inp, outputs=d1, name='embedding')


embedding = make_model_embedding()


# Siamese L1 Distance class
class L1Dist(Layer):

    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


def make_siamese():

    # input images
    input_img = Input(name='inp_img', shape=(105, 105, 3))
    validation_img = Input(name='val_img', shape=(105, 105, 3))

    siamese_lr = L1Dist()
    siamese_lr._name = 'dist'
    # input image and validation image passed through embedding function
    distances = siamese_lr(embedding(input_img), embedding(validation_img))

    cl_lr = Dense(1, activation="sigmoid")(distances)

    return Model(inputs=[input_img, validation_img], outputs=cl_lr)


siamese = make_siamese()
print(siamese.summary())
