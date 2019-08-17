from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib.pyplot as plt
from InverseFaceNetEncoder import InverseFaceNetEncoder
from LoadDataset import LoadDataset
from FaceNet3D import FaceNet3D as Helpers
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf.compat.v1.enable_eager_execution()
print("\n\n\n\nGPU Available:", tf.test.is_gpu_available())
print("\n\n\n\n")


class EncoderTrain(Helpers):
    """ Main function for InverseFaceNet CNN"""
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    def __init__(self):
        super().__init__()

        self.history_list = list()
        self.inverseNet = InverseFaceNetEncoder()

    def training_phase_1(self):
        # Build and compile model:

        self.inverseNet.compile()
        model = self.inverseNet.model
        with tf.device('/device:CPU:0'):
            keras_ds = LoadDataset().load_dataset_batches(_case=self._case)
            keras_ds = keras_ds.shuffle(self.SHUFFLE_BUFFER_SIZE).repeat().batch(
                self.BATCH_SIZE).prefetch(buffer_size=self.AUTOTUNE)

        steps_per_epoch = tf.math.ceil(self.SHUFFLE_BUFFER_SIZE / self.BATCH_SIZE).numpy()
        print("Training with %d steps per epoch" % steps_per_epoch)
        # with tf.device('/device:CPU:0'):
        history_1 = model.fit(keras_ds, epochs=40, steps_per_epoch=steps_per_epoch,
                              callbacks=[self.batch_stats_callback, self.cp_callback])

        self.history_list.append(history_1)

    def plots(self):
        for i in range(0, len(self.history_list)):
            plt.figure()
            plt.ylabel("Custom Loss, phase % d" % i)
            plt.xlabel("Training Steps")
            plt.plot(self.batch_stats_callback.batch_losses)
            plt.savefig(self.plot_path + 'batch_stats_%d.pdf' % i)

            plt.figure()
            plt.title('Mean Squared Error, phase %d' % i)
            plt.plot(self.history_list[i].history['mean_squared_error'])
            plt.savefig(self.plot_path + 'mse%d.pdf' % i)

            plt.figure()
            plt.title('Mean Absolute Error, phase %d' % i)
            plt.plot(self.history_list[i].history['mean_absolute_error'])
            plt.savefig(self.plot_path + 'mae%d.pdf' % i)

            plt.figure()
            plt.title('Loss, phase %d' % i)
            plt.plot(self.history_list[i].history['loss'])
            plt.savefig(self.plot_path + 'loss%d.pdf' % i)


def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(len(gpus), "Physical GPUs")
    # if gpus:
    #     # Restrict TensorFlow to only use the first GPU
    #     try:
    #         tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    #     except RuntimeError as e:
    #         # Visible devices must be set before GPUs have been initialized
    #         print(e)

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
