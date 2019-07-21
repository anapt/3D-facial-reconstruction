from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import keras
from keras import backend as K
import numpy as np
tf.compat.v1.enable_eager_execution()

# PARAMETERS
IMG_SHAPE = (240, 240, 3)
WEIGHT_DECAY = 0.001
BASE_LEARNING_RATE = 0.01

BATCH_SIZE = 20
BATCH_ITERATIONS = 75000

SHUFFLE_BUFFER_SIZE = 1000

# Parameters for Loss
PATH = './DATASET/model2017-1_bfm_nomouth.h5'
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


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

print(tf.executing_eagerly())


def statistical_regularization_term(x):
    tf.compat.v1.enable_eager_execution()
    print(tf.executing_eagerly())
    # sess = tf.compat.v1.Session()
    with tf.compat.v1.Session():
        x = x.eval()


    # x.numpy()
    print(x)
    weight_expression = 0.8
    weight_reflectance = 1.7e-3
    sr_term = K.sum(tf.map_fn(lambda t: t * t, x['shape']), axis=0) \
              + weight_expression * K.sum(tf.map_fn(lambda t: t * t, x['expression'])) \
              + weight_reflectance * K.sum(tf.map_fn(lambda t: t * t, x['reflectance']))

    # print("statistical reg error ", sr_term)
    return sr_term

def model_loss():
    """" Wrapper function which calculates auxiliary values for the complete loss function.
     Returns a *function* which calculates the complete loss given only the input and target output """

    # Photometric alignment Loss
    # photo_loss_func = dense_photometric_alignment
    # # Regularization Loss
    reg_loss_func = statistical_regularization_term

    original_image = model.input
    tensor = tf.multiply(original_image, 1)

    def custom_loss(y_true, y_pred):
        tf.compat.v1.enable_eager_execution()
        print(tf.executing_eagerly())

        original_image = model.input
        original_image = tf.compat.v1.squeeze(original_image, 0)
        # original_image = to_numpy(original_image)
        print("type original image", type(original_image))

        print(type(original_image))

        # Regularization Loss
        reg_loss = reg_loss_func(y_pred)
        # Photometric alignment loss
        print("photo loss")
        # photo_loss = photo_loss_func(x, original_image)

        weight_photo = 1.92
        weight_reg = 2.9e-5

        model_loss = weight_photo * photo_loss + weight_reg * reg_loss
        # model_loss = weight_reg * reg_loss

        return model_loss

    return custom_loss


""" Compiles the Keras model. Includes metrics to differentiate between 
the two main loss terms """
model.compile(optimizer=tf.keras.optimizers.Adadelta(lr=BASE_LEARNING_RATE,
                                                     rho=0.95, epsilon=None, decay=0.0),
              loss=model_loss(),
              metrics=['mean_absolute_error'])
print('Model Compiled!')
