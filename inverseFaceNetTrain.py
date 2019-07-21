from __future__ import absolute_import, division, print_function

import tensorflow as tf

tf.compat.v1.enable_eager_execution()

import matplotlib.pyplot as plt
from InverseFaceNet import InverseFaceNetModel
from loadDataset import load_dataset
import CollectBatchStats as batch_stats


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

    batch_stats_callback = batch_stats.CollectBatchStats()
    model.fit(keras_ds, epochs=10, steps_per_epoch=steps_per_epoch, callbacks=[batch_stats_callback, cp_callback])

    plt.figure()
    plt.ylabel("Loss")
    plt.xlabel("Training Steps")
    # plt.ylim([0,2])
    plt.plot(batch_stats_callback.batch_losses)
    plt.show()


main()
