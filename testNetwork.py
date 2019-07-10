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
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')