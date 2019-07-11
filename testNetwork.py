from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

keras = tf.keras

''' Parameters '''
IMG_SIZE = 240
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model AlexNet
# keras.applications.xception.Xception(include_top=True,
#                                      weights='imagenet', input_tensor=None,
#                                      input_shape=None, pooling=None, classes=1000)

# 75 k batch iteration with batch size of 32
# weight decay of 0.001 l2 regularizer
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

base_model = tf.keras.applications.xception.Xception(include_top=False,
                                                     weights='imagenet',
                                                     input_tensor=None,
                                                     input_shape=IMG_SHAPE)
base_model.trainable = False
# base_model.summary()

weights_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
prediction_layer = tf.keras.layers.Dense(257, activation=None, use_bias=True,
                                         kernel_initializer=weights_init, bias_initializer='zeros')

base_learning_rate = 0.01
# base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
#                                                include_top=False,
#                                                weights='imagenet')

print("Number of layers in the base model: ", len(base_model.layers))
model = tf.keras.Sequential([
  base_model,
  prediction_layer
])

model.summary()

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

