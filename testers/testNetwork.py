from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from loadDataset import load_dataset
from customLoss import custom_loss
from keras import backend as K

AUTOTUNE = tf.data.experimental.AUTOTUNE


keras = tf.keras

''' Parameters '''
IMG_SIZE = 240
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

WEIGHT_DECAY = 0.001
BASE_LEARNING_RATE = 0.01

BATCH_SIZE = 20
BATCH_ITERATIONS = 75000

SHUFFLE_BUFFER_SIZE = 1000


def create_model():

    # Create the base model from the pre-trained model XCEPTION
    base_model = tf.keras.applications.xception.Xception(include_top=False,
                                                         weights='imagenet',
                                                         input_tensor=None,
                                                         input_shape=IMG_SHAPE)
    base_model.trainable = False
    # base_model.summary()
    weights_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

    # Create global average pooling layer
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    # Create prediction layer
    prediction_layer = tf.keras.layers.Dense(257, activation=None, use_bias=True,
                                             kernel_initializer=weights_init, bias_initializer='zeros',
                                             kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))

    # Code to check compatibility of dimensions
    # for image_batch, label_batch in keras_ds.take(1):
    #     pass
    # print('image batch shape ', image_batch.shape)
    # feature_batch = base_model(image_batch)
    # print('feature batch shape', feature_batch.shape)
    # feature_batch_average = global_average_layer(feature_batch)
    # print(feature_batch_average.shape)
    # prediction_batch = prediction_layer(feature_batch_average)
    # print(prediction_batch.shape)

    # print("Number of layers in the base model: ", len(base_model.layers))

    # Stack model layers
    model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        prediction_layer
    ])

    # model.summary()
    # print("Number of layers in the model: ", len(model.layers))

    # Compile model
    model.compile(optimizer=keras.optimizers.Adadelta(lr=BASE_LEARNING_RATE,
                                                      rho=0.95, epsilon=None, decay=0.0),
                  loss=custom_loss(base_model.layers[0].output, prediction_layer.output),
                  metrics=['mean_absolute_error'])

    return model


model = create_model()

keras_ds = load_dataset()
keras_ds = keras_ds.shuffle(SHUFFLE_BUFFER_SIZE).repeat().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

steps_per_epoch = tf.math.ceil(SHUFFLE_BUFFER_SIZE/BATCH_SIZE).numpy()
print("Training with %d steps per epoch" % steps_per_epoch)

# Create callbacks path and dir
checkpoint_path = "./DATASET/training/cp-{epoch:04d}.ckpt"
checkpoint_dir = "./DATASET/training/"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=5)

model.fit(keras_ds, epochs=1, steps_per_epoch=steps_per_epoch, callbacks=[cp_callback])

# latest = tf.train.latest_checkpoint(checkpoint_dir)
