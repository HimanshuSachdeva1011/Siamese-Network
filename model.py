import os
from keras import optimizers, losses
from keras.models import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
from keras.utils import Progbar
from keras.metrics import Precision, Recall

# GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
# Ignores those Big GPU Messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# define anchor and positive path
anchor_path = os.path.join('data', 'anchor')
positive_path = os.path.join('data', 'positive')
negative_path = os.path.join('data', 'negative')

# load dataset
anchor = tf.data.Dataset.list_files(anchor_path+'\*.jpg').take(291)
positive = tf.data.Dataset.list_files(positive_path+'\*.jpg').take(273)
negative = tf.data.Dataset.list_files(negative_path+'\*.jpg').take(273)


def preprocess(file_path):

    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    # Resizing image to 105x105x3
    img = tf.image.resize(img, (105, 105))
    # Scaling image b/w 0 and 1
    img = img / 255.0

    return img


# pos_ve = labelled ones of anchor length
# neg_ve = labelled zeros of anchor length
pos_ve = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
neg_ve = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))

# Concatenate files to data variable
data = pos_ve.concatenate(neg_ve)


# train test partition
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

# embed: inp->conv(64,(10,10))+max-pool(64,(2,2))->conv(128,(4,4)+max-pool(128,(2,2)) -> conv(256,(4,4)) -> dense(4096)


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


# embedding ↘
#            distance_layer → output
# embedding ↗
def make_siamese_model():

    # input images
    input_img = Input(name='inp_img', shape=(105, 105, 3))
    validation_img = Input(name='val_img', shape=(105, 105, 3))

    siamese_lr = L1Dist()
    siamese_lr._name = 'dist'
    # input image and validation image passed through embedding function
    distances = siamese_lr(embedding(input_img), embedding(validation_img))
    # dense output layer for result
    op_lr = Dense(1, activation="sigmoid")(distances)

    return Model(inputs=[input_img, validation_img], outputs=op_lr)


siamese_model = make_siamese_model()

opt = optimizers.Adam(1e-4)
binary_loss = losses.BinaryCrossentropy()

siamese_model.compile(
    optimizer=opt,
    loss=binary_loss,
    metrics=["accuracy"]
)


@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        # image
        x = batch[:2]
        # label
        y = batch[2]

        # Forward pass
        yhat = siamese_model(x, training=True)
        # Calculate loss
        loss = binary_loss(y, yhat)

    print(loss)

    # Calculate gradient (Forward Pass)
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    # Updated weights applied to siamese model (Backward pass)
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

    # Return loss
    return loss


def train(dat, tot_epochs):
    # Loop through epochs
    for epochs in range(1, tot_epochs + 1):
        print('Epoch {}/{}'.format(epochs, tot_epochs))
        progbar = Progbar(len(dat))

        # Loop through each batch
        for idx, batch in enumerate(dat):
            # Run train step here
            train_step(batch)
            progbar.update(idx + 1)


# bombs away
epoch = 55
train(train_data, epoch)

# accuracy and other metrics
test_input, test_val, y_true = test_data.as_numpy_iterator().next()
y_hat = siamese_model.predict([test_input, test_val])

# append results to array
rec = Recall()
rec.update_state(y_true, y_hat)
print(rec.result().numpy())

prec_ = Precision()
prec_.update_state(y_true, y_hat)
print(prec_.result().numpy())

# Save Model file
siamese_model.save("siamese.h5")
