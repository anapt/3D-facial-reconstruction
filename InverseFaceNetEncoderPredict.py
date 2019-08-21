from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from InverseFaceNetEncoder import InverseFaceNetEncoder
from LoadDataset import LoadDataset
from ImageFormationLayer import ImageFormationLayer
import numpy as np
import matplotlib.pyplot as plt
from FaceNet3D import FaceNet3D as Helpers
from prediction_plots import prediction_plots

tf.compat.v1.enable_eager_execution()


class InverseFaceNetEncoderPredict(Helpers):
    def __init__(self):
        super().__init__()
        self.latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        # self.latest = "./DATASET/training/cp-0040.ckpt"
        print("Latest checkpoint: ", self.latest)
        self.encoder = InverseFaceNetEncoder()
        self.model = self.load_model()

    def load_model(self):
        """
        Load trained model and compile

        :return: Compiled Keras model
        """
        self.encoder.build_model()
        model = self.encoder.model
        model.load_weights(self.latest)

        self.encoder.compile()
        # model = self.encoder.model

        return model

    def evaluate_model(self):
        """ Evaluate model on validation data """
        test_ds = LoadDataset().load_dataset_single_image()
        loss, mse, mae = self.model.evaluate(test_ds)
        print("\nRestored model, Loss: {0} \nMean Squared Error: {1}\n"
              "Mean Absolute Error: {2}\n".format(loss, mse, mae))

    def model_predict(self, image_path):

        image = LoadDataset().load_and_preprocess_image_4d(image_path)
        x = self.model.predict(image)

        return np.transpose(x)

    @staticmethod
    def calculate_decoder_output(x):
        """
        Reconstruct image

        :param x: <class 'numpy.ndarray'> with shape (257, ) : semantic code vector
        :return: <class 'numpy.ndarray'> with shape (240, 240, 3)
        """
        decoder = ImageFormationLayer(x)

        image = decoder.get_reconstructed_image()

        return image


def main():
    net = InverseFaceNetEncoderPredict()
    n = 0
    # net.evaluate_model()
    image_path = net.data_root + 'training/image_{:06}.png'.format(n)

    x = net.model_predict(image_path)
    x = net.vector2dict(x)
    x_true = np.loadtxt(net.sem_root + 'training/x_{:06}.txt'.format(0))
    x_true = net.vector2dict(x_true)
    prediction_plots(x_true, x, save_figs=False)

    image = net.calculate_decoder_output(x)

    show_result = True
    if show_result:
        plt.imshow(image)
        plt.show()


main()
