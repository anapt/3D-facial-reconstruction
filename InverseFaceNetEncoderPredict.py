from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from InverseFaceNetEncoder import InverseFaceNetEncoder
from LoadDataset import LoadDataset
from ImageFormationLayer import ImageFormationLayer
import numpy as np
import matplotlib.pyplot as plt
from FaceNet3D import FaceNet3D as Helpers
from prediction_plots import prediction_plots
import cv2
import pathlib
tf.compat.v1.enable_eager_execution()


class InverseFaceNetEncoderPredict(Helpers):
    def __init__(self):
        """
        Class initializer
        """
        super().__init__()
        # self.latest = self.trained_models_dir + "cp-0360.ckpt"
        self.latest = "/home/anapt/data/until_convergence/cp-0205.ckpt"
        # self.latest = self.checkpoint_dir + "cp-7500.ckpt"
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
        """
        Evaluate model on validation data
        """
        with tf.device('/device:CPU:0'):
            test_ds = LoadDataset().load_dataset_single_image(self._case)
            loss, mse, mae = self.model.evaluate(test_ds)
            print("\nRestored model, Loss: {0} \nMean Squared Error: {1}\n"
                  "Mean Absolute Error: {2}\n".format(loss, mse, mae))

    def model_predict(self, image_path):
        """
        Predict out of image_path
        :param image_path: path
        :return:
        """
        image = LoadDataset().load_and_preprocess_image_4d(image_path)
        x = self.model.predict(image)

        return np.transpose(x)

    @staticmethod
    def calculate_decoder_output(x):
        """
        Reconstruct image

        :param x: <class 'numpy.ndarray'> with shape (self.scv_length, ) : semantic code vector
        :return: <class 'numpy.ndarray'> with shape self.IMG_SHAPE
        """
        decoder = ImageFormationLayer(x)

        image = decoder.get_reconstructed_image()

        return image
