from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from testImport2 import load_dataset
AUTOTUNE = tf.data.experimental.AUTOTUNE


keras = tf.keras

''' Parameters '''
IMG_SIZE = 240
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

WEIGHT_DECAY = 0.001
BASE_LEARNING_RATE = 0.01

BATCH_SIZE = 10
BATCH_ITERATIONS = 75000

SHUFFLE_BUFFER_SIZE = 100

# Create the base model from the pre-trained model XCEPTION
base_model = tf.keras.applications.xception.Xception(include_top=False,
                                                     weights='imagenet',
                                                     input_tensor=None,
                                                     input_shape=IMG_SHAPE)
base_model.trainable = False
# base_model.summary()

weights_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
prediction_layer = tf.keras.layers.Dense(257, activation=None, use_bias=True,
                                         kernel_initializer=weights_init, bias_initializer='zeros',
                                         kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))


keras_ds = load_dataset()
keras_ds = keras_ds.shuffle(SHUFFLE_BUFFER_SIZE).repeat().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

for image_batch, label_batch in keras_ds.take(1):
  pass

# print('image batch shape ', image_batch.shape)

feature_batch = base_model(image_batch)
# print('feature batch shape', feature_batch.shape)

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
# print(feature_batch_average.shape)

prediction_batch = prediction_layer(feature_batch_average)
# print(prediction_batch.shape)
# print("Number of layers in the base model: ", len(base_model.layers))

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

# model.summary()
# print("Number of layers in the model: ", len(model.layers))

model.compile(optimizer=keras.optimizers.Adadelta(lr=BASE_LEARNING_RATE,
                                                  rho=0.95, epsilon=None, decay=0.0),
              loss='mean_squared_error',
              metrics=['accuracy'])

steps_per_epoch = tf.math.ceil(SHUFFLE_BUFFER_SIZE/BATCH_SIZE).numpy()
# print(steps_per_epoch)
#
model.fit(keras_ds, epochs=10, steps_per_epoch=steps_per_epoch)
