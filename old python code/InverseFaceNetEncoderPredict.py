from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from old_encoder import InverseFaceNetEncoder
from loadDataset import load_dataset_single_image, load_and_preprocess_image_4d
import ImageFormationLayer as ifl
import numpy as np
import SemanticCodeVector as scv
import matplotlib.pyplot as plt

tf.compat.v1.enable_eager_execution()


class InverseFaceNetEncoderPredict(object):
    def __init__(self):
        self.checkpoint_dir = "/home/anapt/data/training/trained_10_105/"
        self.latest = tf.train.latest_checkpoint(self.checkpoint_dir)
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
        test_ds = load_dataset_single_image()
        loss, mse, mae = self.model.evaluate(test_ds)
        print("\nRestored model, Loss: {0} \nMean Squared Error: {1}\n"
              "Mean Absolute Error: {2}\n".format(loss, mse, mae))

    def model_predict(self, image_path):

        image = load_and_preprocess_image_4d(image_path)
        x = self.model.predict(image)

        return np.transpose(x)

    @staticmethod
    def calculate_decoder_output(x):
        """
        Reconstruct image

        :param x: <class 'numpy.ndarray'> with shape (257, ) : semantic code vector
        :return: <class 'numpy.ndarray'> with shape (240, 240, 3)
        """
        decoder = ifl.ImageFormationLayer(x)

        image = decoder.get_reconstructed_image()

        return image


def main():
    net = InverseFaceNetEncoderPredict()

    # net.evaluate_model()
    image_path = './DATASET/images/validation/image_{:06}.png'.format(1)

    x = net.model_predict(image_path)
    np.savetxt("./x_boot.txt", x)

    image = net.calculate_decoder_output(x)

    show_result = True
    if show_result:
        plt.imshow(image)
        plt.show()


main()
