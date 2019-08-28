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
        super().__init__()
        self.latest = self.trained_models_dir + "cp-15000.ckpt"
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

    # def evaluate_model(self):
    #     """ Evaluate model on validation data """
    #     test_ds = LoadDataset().load_dataset_single_image()
    #     loss, mse, mae = self.model.evaluate(test_ds)
    #     print("\nRestored model, Loss: {0} \nMean Squared Error: {1}\n"
    #           "Mean Absolute Error: {2}\n".format(loss, mse, mae))

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
    # n = 11
    # path = net.bootstrapping_path + 'MUG/'
    # data_root = pathlib.Path(path)
    # all_image_paths = list(data_root.glob('*.png'))
    # all_image_paths = [str(path) for path in all_image_paths]
    # all_image_paths.sort()
    # print(all_image_paths)
    # all_image_paths = all_image_paths[5:16]
    # for n, path in enumerate(all_image_paths):
    #
    #     # net.evaluate_model()
    #
    #     x = net.model_predict(path)
    #     x = net.vector2dict(x)
    #     # x_true = np.loadtxt(net.sem_root + 'training/x_{:06}.txt'.format(n))
    #     # x_true = np.zeros((net.scv_length, ))
    #     # x_true = net.vector2dict(x_true)
    #
    #     # prediction_plots(x_true, x, save_figs=False)
    #
    #     # loss = np.power(net.dict2vector(x) - net.dict2vector(x_true), 2)
    #     # print("Loss: {}".format(np.mean(loss)))
    #     image = net.calculate_decoder_output(x)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     cv2.imwrite(net.bootstrapping_path + 'images/image0_{:06}.png'.format(n), image)
    x = np.loadtxt("/home/anapt/data/semantic/x_{:06}.txt".format(8))
    # x = np.loadtxt("./DATASET/semantic/training/x_{:06}.txt".format(1))
    x = net.vector2dict(x)
    image = net.calculate_decoder_output(x)
    show_result = True
    if show_result:
        plt.imshow(image)
        plt.show()


# main()
