from __future__ import absolute_import, division, print_function, unicode_literals


import argparse
import json
import os
import sys
import tensorflow as tf

from InverseFaceNet import InverseFaceNetModel
from loadDataset import load_dataset


def main():
    """ Main function for InverseFaceNet CNN"""
    # Parameters
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    BATCH_SIZE = 20
    SHUFFLE_BUFFER_SIZE = 1000

    # Build and compile model:
    inverseNet = InverseFaceNetModel()
    inverseNet.compile()
    model = inverseNet.model

    checkpoint_path = "./DATASET/training/cp-{epoch:04d}.ckpt"
    checkpoint_dir = "./DATASET/training/"

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True,
        # Save weights, every 5-epochs.
        period=5)

    keras_ds = load_dataset()
    keras_ds = keras_ds.shuffle(SHUFFLE_BUFFER_SIZE).repeat().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

    steps_per_epoch = tf.math.ceil(SHUFFLE_BUFFER_SIZE / BATCH_SIZE).numpy()
    print("Training with %d steps per epoch" % steps_per_epoch)

    model.fit(keras_ds, epochs=1, steps_per_epoch=steps_per_epoch, callbacks=[cp_callback])

    # latest = tf.train.latest_checkpoint(checkpoint_dir)


main()
