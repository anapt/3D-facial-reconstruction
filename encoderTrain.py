from __future__ import absolute_import, division, print_function

import tensorflow as tf
import matplotlib.pyplot as plt
from InverseFaceNetEncoder import InverseFaceNetEncoder
from loadDataset import load_dataset_batches
import CollectBatchStats as batch_stats

tf.compat.v1.enable_eager_execution()


class EncoderTrain:
    """ Main function for InverseFaceNet CNN"""
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    def __init__(self):
        # Parameters
        self.checkpoint_dir = "./DATASET/training100/"
        self.checkpoint_path = "./DATASET/training100/cp-{epoch:04d}.ckpt"

        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(
            self.checkpoint_path, verbose=1, save_weights_only=True,
            # Save weights, every 5-epochs.
            period=10)

        self.batch_stats_callback = batch_stats.CollectBatchStats()

        self.history_list = list()

        self.inverseNet = InverseFaceNetEncoder()
        self.BATCH_SIZE = self.inverseNet.BATCH_SIZE
        self.SHUFFLE_BUFFER_SIZE = self.inverseNet.SHUFFLE_BUFFER_SIZE

    def training_phase_1(self):
        # Build and compile model:

        self.inverseNet.compile()
        model = self.inverseNet.model

        keras_ds = load_dataset_batches(_case='training')
        keras_ds = keras_ds.shuffle(self.SHUFFLE_BUFFER_SIZE).repeat().batch(self.BATCH_SIZE).prefetch(buffer_size=
                                                                                                       self.AUTOTUNE)

        steps_per_epoch = tf.math.ceil(self.SHUFFLE_BUFFER_SIZE / self.BATCH_SIZE).numpy()
        print("Training with %d steps per epoch" % steps_per_epoch)

        history_1 = model.fit(keras_ds, epochs=10, steps_per_epoch=steps_per_epoch,
                              callbacks=[self.batch_stats_callback, self.cp_callback])

        self.history_list.append(history_1)

    def training_phase_2(self):
        # load weights trained on synthetic faces and start bootstrapping
        latest = tf.train.latest_checkpoint(self.checkpoint_dir)

        model = self.inverseNet.model
        model.load_weights(latest)

        self.inverseNet.compile()

        bootstrapping_ds = load_dataset_batches(_case='training')
        bootstrapping_ds = bootstrapping_ds.shuffle(self.SHUFFLE_BUFFER_SIZE).repeat().\
            batch(self.BATCH_SIZE).prefetch(buffer_size=self.AUTOTUNE)

        steps_per_epoch = tf.math.ceil(self.SHUFFLE_BUFFER_SIZE / self.BATCH_SIZE).numpy()
        print("Training with %d steps per epoch" % steps_per_epoch)

        bootstrapping = model.fit(bootstrapping_ds, epochs=90, steps_per_epoch=steps_per_epoch,
                                  callbacks=[self.batch_stats_callback, self.cp_callback])

        self.history_list.append(bootstrapping)

    def plots(self):
        for i in range(0, len(self.history_list)):
            plt.figure()
            plt.ylabel("Custom Loss, phase % d" % i)
            plt.xlabel("Training Steps")
            plt.plot(self.batch_stats_callback.batch_losses)
            plt.savefig('batch_stats_%d.pdf' % i)

            plt.figure()
            plt.title('Mean Squared Error, phase %d' % i)
            plt.plot(self.history_list[i].history['mean_squared_error'])
            plt.savefig('mse%d.pdf' % i)

            plt.figure()
            plt.title('Mean Absolute Error, phase %d' % i)
            plt.plot(self.history_list[i].history['mean_absolute_error'])
            plt.savefig('mae%d.pdf' % i)

            plt.figure()
            plt.title('Loss, phase %d' % i)
            plt.plot(self.history_list[i].history['loss'])
            plt.savefig('loss%d.pdf' % i)


def main():

    train = EncoderTrain()
    print("\n \n \nPhase 1\n START")

    train.training_phase_1()

    print("Phase 1: COMPLETE")
    print("\n \n \nPhase 2\n START")

    # train.training_phase_2()

    print("Phase 2: COMPLETE")
    print("Saving plots...")

    train.plots()


main()
