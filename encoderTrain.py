from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib.pyplot as plt
from InverseFaceNetEncoder import InverseFaceNetEncoder
from loadDataset import load_dataset_batches
import CollectBatchStats as batch_stats
from keras import backend as K

tf.compat.v1.enable_eager_execution()
print("\n\n\n\nGPU Available:", tf.test.is_gpu_available())
print("\n\n\n\n")


class EncoderTrain:
    """ Main function for InverseFaceNet CNN"""
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    def __init__(self):
        # Parameters
        self.checkpoint_dir = "./DATASET/training/"
        self.checkpoint_path = "./DATASET/training/cp-2-{epoch:04d}.ckpt"

        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(
            self.checkpoint_path, verbose=1, save_weights_only=True,
            # Save weights, every 5-epochs.
            period=1)

        self.batch_stats_callback = batch_stats.CollectBatchStats()

        self.history_list = list()

        self.inverseNet = InverseFaceNetEncoder()
        self.BATCH_SIZE = self.inverseNet.BATCH_SIZE
        self.SHUFFLE_BUFFER_SIZE = self.inverseNet.SHUFFLE_BUFFER_SIZE

    def training_phase_1(self):
        # Build and compile model:

        self.inverseNet.compile()
        model = self.inverseNet.model
        with tf.device('/device:CPU:0'):
            keras_ds = load_dataset_batches(_case='training')
            keras_ds = keras_ds.shuffle(self.SHUFFLE_BUFFER_SIZE).repeat().batch(
                self.BATCH_SIZE).prefetch(buffer_size=self.AUTOTUNE)

        steps_per_epoch = tf.math.ceil(self.SHUFFLE_BUFFER_SIZE / self.BATCH_SIZE).numpy()
        print("Training with %d steps per epoch" % steps_per_epoch)

        with tf.device('/device:CPU:0'):
            history_1 = model.fit(keras_ds, epochs=10, steps_per_epoch=steps_per_epoch,
                                  callbacks=[self.batch_stats_callback, self.cp_callback])

        self.history_list.append(history_1)

    def training_phase_12(self):
        # Build and compile model:

        # load weights trained on synthetic faces and start bootstrapping
        latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        print("\n \n \n\n checkpoint: ", latest)
        print("\n\n\n\n\n\n")
        model = self.inverseNet.model
        model.load_weights(latest)

        self.inverseNet.compile()

        with tf.device('/device:CPU:0'):
            keras_ds = load_dataset_batches(_case='training')
            keras_ds = keras_ds.shuffle(self.SHUFFLE_BUFFER_SIZE).repeat().batch(
                self.BATCH_SIZE).prefetch(buffer_size=self.AUTOTUNE)

        steps_per_epoch = tf.math.ceil(self.SHUFFLE_BUFFER_SIZE / self.BATCH_SIZE).numpy()
        print("Training with %d steps per epoch" % steps_per_epoch)

        with tf.device('/device:CPU:0'):
            history_1 = model.fit(keras_ds, epochs=9, steps_per_epoch=steps_per_epoch,
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
            plt.savefig('./plots/batch_stats_%d.pdf' % i)

            plt.figure()
            plt.title('Mean Squared Error, phase %d' % i)
            plt.plot(self.history_list[i].history['mean_squared_error'])
            plt.savefig('./plots/mse%d.pdf' % i)

            plt.figure()
            plt.title('Mean Absolute Error, phase %d' % i)
            plt.plot(self.history_list[i].history['mean_absolute_error'])
            plt.savefig('./plots/mae%d.pdf' % i)

            plt.figure()
            plt.title('Loss, phase %d' % i)
            plt.plot(self.history_list[i].history['loss'])
            plt.savefig('./plots/loss%d.pdf' % i)


def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.debugging.set_log_device_placement(True)
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    #     try:
    #         tf.config.experimental.set_virtual_device_configuration(
    #             gpus[0],
    #             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*4)])
    #         tf.config.experimental.set_virtual_device_configuration(
    #             gpus[1],
    #             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 4)])
    #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Virtual devices must be set before GPUs have been initialized
    #         print(e)
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    train = EncoderTrain()
    print("\n \n \nPhase 1\nSTART")

    with tf.device('/device:CPU:0'):
        train.training_phase_1()

    print("Phase 1: COMPLETE")
    # print("\n \n \nPhase 2\n START")
    #
    # # train.training_phase_2()
    #
    # print("Phase 2: COMPLETE")
    print("Saving plots...")

    train.plots()


main()
