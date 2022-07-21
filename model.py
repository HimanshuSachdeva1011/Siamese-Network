import os
import keras.activations
from keras import losses, optimizers
from keras.models import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
from tensorflow import train
import preprocess

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def make_model():
    inp = Input(shape=(100, 100, 3), name='input_image')

    conv1 = Conv2D(64, (10, 10), activation='relu')(inp)
    max1 = MaxPooling2D(64, (2, 2), padding='same')(conv1)

    conv2 = Conv2D(128, (7, 7), activation='relu')(max1)
    m2 = MaxPooling2D(64, (2, 2), padding='same')(conv2)

    conv3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding='same')(conv3)

    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=[inp], outputs=[d1], name='embedding')


embedding = make_model()


# Siamese L1 Distance class
class L1Dist(Layer):

    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


def make_siamese():

    # input images
    input_img = Input(name='ip_img', shape=(100, 100, 3))
    validation_img = Input(name='val_img', shape=(100, 100, 3))

    siamese_lr = L1Dist()
    siamese_lr._name = 'distance'
    # input image and validation image passed through embedding function
    distances = siamese_lr(embedding(input_img), embedding(validation_img))

    cl_lr = Dense(1, activation="sigmoid")(distances)

    return Model(inputs=[input_img, validation_img], outputs=cl_lr)


siamese = make_siamese()
print(siamese.summary())

# loss and optimizer
losses = losses.BinaryCrossentropy()
opt = keras.optimizers.Adam(learning_rate=1e-4)

check_pt_dr = './training_checkpoints'
check_pt_prefix = os.path.join(check_pt_dr, 'ckpt')
check_pt = train.Checkpoint(optimizer=opt, siamese=siamese)


@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        x = batch[:2]
        y = batch[2]

        # Forward Pass
        yhat = siamese(x, training=True)
        # Calc Loss
        loss = losses(y, yhat)
    print(loss)

    grad = tape.gradient(loss, siamese.trainable_variables)

    opt.apply_gradients(zip(grad, siamese.trainable_variables))

    return loss


# Train
def train(data, epochs=50):
    for epoch in range(0, epochs):
        print('\n Epoch {}/{}'.format(epoch, epochs))
        progbar = tf.keras.utils.Progbar(len(data))

    # Loop through each batch
    for idx, batch in enumerate(data):
        # Run train step here
        train_step(batch)
        progbar.update(idx + 1)

    # Save checkpoints
    if epoch % 10 == 0:
        check_pt.save(file_prefix=check_pt_prefix)

