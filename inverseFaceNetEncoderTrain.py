from __future__ import absolute_import, division, print_function

import tensorflow as tf
import matplotlib.pyplot as plt
from InverseFaceNetEncoder import InverseFaceNetEncoder
from loadDataset import load_training_dataset
import CollectBatchStats as batch_stats

tf.compat.v1.enable_eager_execution()


def main():
    """ Main function for InverseFaceNet CNN"""
    # Parameters
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # Build and compile model:
    inverseNet = InverseFaceNetEncoder()
    BATCH_SIZE = inverseNet.BATCH_SIZE
    SHUFFLE_BUFFER_SIZE = inverseNet.SHUFFLE_BUFFER_SIZE
    inverseNet.compile()
    model = inverseNet.model

    checkpoint_path = "./DATASET/training/cp-{epoch:04d}.ckpt"

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True,
        # Save weights, every 5-epochs.
        period=5)

    keras_ds = load_training_dataset()
    keras_ds = keras_ds.shuffle(SHUFFLE_BUFFER_SIZE).repeat().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

    steps_per_epoch = tf.math.ceil(SHUFFLE_BUFFER_SIZE / BATCH_SIZE).numpy()
    print("Training with %d steps per epoch" % steps_per_epoch)

    batch_stats_callback = batch_stats.CollectBatchStats()
    history = model.fit(keras_ds, epochs=40, steps_per_epoch=8, callbacks=[batch_stats_callback, cp_callback])

    model.save('my_model.h5')

    plt.figure()
    plt.ylabel("Loss")
    plt.xlabel("Training Steps")
    plt.plot(batch_stats_callback.batch_losses)
    plt.savefig('batch_stats.pdf')

    plt.figure()
    plt.title('Mean Squared Error')
    plt.plot(history.history['mean_squared_error'])
    plt.savefig('mse.pdf')

    plt.figure()
    plt.title('Mean Absolute Error')
    plt.plot(history.history['mean_absolute_error'])
    plt.savefig('mae.pdf')

    plt.figure()
    plt.title('Loss')
    plt.plot(history.history['loss'])
    plt.savefig('loss.pdf')


main()
