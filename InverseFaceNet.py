from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from keras import backend as K
from InverseFaceNetEncoder import InverseFaceNetEncoder
from loadDataset import load_testing_dataset, load_and_preprocess_image_4d
import ImageFormationLayer as ifl
import numpy as np
import SemanticCodeVector as scv
import matplotlib.pyplot as plt

tf.compat.v1.enable_eager_execution()


class InverseFaceNet(object):
    def __init__(self):
        self.PATH_DIR = './DATASET/model/'
        self.PATH = self.PATH_DIR + 'model16im.h5'
        self.checkpoint_dir = "./DATASET/training/"
        self.latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        print(self.latest)
        self.encoder = InverseFaceNetEncoder()
        self.model = self.load_model()

    def load_model(self):
        self.encoder.build_model()
        model = self.encoder.model
        model.load_weights(self.latest)

        self.encoder.compile()
        # model = self.encoder.model

        return model

    def evaluate_model(self):

        test_ds = load_testing_dataset()
        loss, mse, mae = self.model.evaluate(test_ds)
        print("Restored model, Loss: {0}, Mean Squared Error: {1}, Mean Absolute Error: {2}".format(loss, mse, mae))

    def model_predict(self, image_path):

        image = load_and_preprocess_image_4d(image_path)
        x = self.model.predict(image)

        return np.transpose(x)

    def calculate_decoder_output(self, x):

        decoder = ifl.ImageFormationLayer(x)

        image = decoder.get_reconstructed_image()

        return image


def main():
    net = InverseFaceNet()

    # net.evaluate_model()
    image_path = './DATASET/images/over/image_{:06}.png'.format(0)

    x = net.model_predict(image_path)
    np.savetxt("./x_pred.txt", x)

    # image = net.calculate_decoder_output(x)
    #
    # show_result = True
    # if show_result:
    #     plt.imshow(image)
    #     plt.show()


main()
