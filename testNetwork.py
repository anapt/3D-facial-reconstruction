from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


keras = tf.keras

''' Parameters '''
IMG_SIZE = 240
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

WEIGHT_DECAY = 0.001
BASE_LEARNING_RATE = 0.01

BATCH_SIZE = 32
BATCH_ITERATIONS = 75000

SHUFFLE_BUFFER_SIZE = 1000

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

print("Number of layers in the base model: ", len(base_model.layers))
model = tf.keras.Sequential([
  base_model,
  prediction_layer
])

model.summary()
print("Number of layers in the model: ", len(model.layers))
model.compile(optimizer=keras.optimizers.Adadelta(lr=BASE_LEARNING_RATE,
                                                  rho=0.95, epsilon=None, decay=0.0),
              loss='mean_squared_error',
              metrics=['accuracy'])

