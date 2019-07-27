from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from keras import backend as K
from InverseFaceNetEncoder import InverseFaceNetEncoder
from loadDataset import load_testing_dataset
import numpy as np
import SemanticCodeVector as scv

tf.compat.v1.enable_eager_execution()


class InverseFaceNet(object):
    def __init__(self):
        self.PATH_DIR = './DATASET/model/'
        self.PATH = self.PATH_DIR + 'model16im.h5'
        self.checkpoint_dir = "./DATASET/training/"
        self.latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        self.encoder = InverseFaceNetEncoder()

    def load_model(self):
        self.encoder.build_model()
        model = self.encoder.model
        model.load_weights(self.latest)

        self.encoder.compile()
        # model = self.encoder.model

        return model


def main():
    net = InverseFaceNet()
    model = net.load_model()

    test_ds = load_testing_dataset()

    loss, mse, mae = model.evaluate(test_ds)
    print("Restored model, Loss: {0}, Mean Squared Error: {1}, Mean Absolute Error: {2}".format(loss, mse, mae))


main()
