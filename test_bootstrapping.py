from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib.pyplot as plt
from InverseFaceNetEncoder import InverseFaceNetEncoder
from loadDataset import load_dataset_batches
import CollectBatchStats as batch_stats
from keras import backend as K

tf.compat.v1.enable_eager_execution()


class Bootstrapping():
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    def __init__(self):
        # Parameters
        self.checkpoint_dir = "/home/anapt//data/models/10_epochs_105"
        self.checkpoint_path = "/home/anapt//data/models/10_epochs_105/cp-3-{epoch:04d}.ckpt"

        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(
            self.checkpoint_path, verbose=1, save_weights_only=True,
            # Save weights, every 5-epochs.
            period=1)

        self.batch_stats_callback = batch_stats.CollectBatchStats()

        self.history_list = list()

        self.inverseNet = InverseFaceNetEncoder()
        self.BATCH_SIZE = self.inverseNet.BATCH_SIZE
        self.SHUFFLE_BUFFER_SIZE = self.inverseNet.SHUFFLE_BUFFER_SIZE

    def training_phase_12(self):
        # Build and compile model:

        # load weights trained on synthetic faces and start bootstrapping
        latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        print("\n \n \n\n checkpoint: ", latest)
        print("\n\n\n\n\n\n")
        model = self.inverseNet.model
        model.load_weights(latest)

        self.inverseNet.compile()

        keras_ds = load_dataset_batches(_case='training')
        keras_ds = keras_ds.shuffle(self.SHUFFLE_BUFFER_SIZE).repeat().batch(
            self.BATCH_SIZE).prefetch(buffer_size=self.AUTOTUNE)


def main():
    train = Bootstrapping()
    train.training_phase_12()


main()
