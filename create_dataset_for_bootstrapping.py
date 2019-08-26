import numpy as np
from ImageFormationLayer import ImageFormationLayer
import matplotlib.pyplot as plt
import pathlib
from InverseFaceNetEncoderPredict import InverseFaceNetEncoderPredict
from FaceNet3D import FaceNet3D as Helpers
import cv2
import time
from prediction_plots import prediction_plots


class PrepareImages(Helpers):

    def __init__(self):
        super().__init__()
        self.image_dir = '/home/anapt/Documents/MUG/cropped/'
        self.net = InverseFaceNetEncoderPredict()

    def vector_resampling(self, vector):
        vector = self.vector2dict(vector)

        shape = vector['shape'] + np.random.normal(0, 0.05, self.shape_dim)

        expression = vector['expression'] + np.random.normal(0, 0.1, self.expression_dim)

        color = vector['color'] + np.random.normal(0, 0.1, self.color_dim)

        rotation = vector['rotation'] + np.random.uniform(-0.5, 0.5, 3)

        x = {
            "shape": shape,
            "expression": expression,
            "color": color,
            "rotation": rotation,
        }

        return x

    def get_prediction(self, image_path):
        vector = self.net.model_predict(image_path=image_path)
        # prediction_plots(self.vector2dict(vector), self.vector2dict(np.zeros(231, )), save_figs=False)
        return vector

    def create_image_and_save(self, vector, n):
        # create first image with variation
        x_new = self.vector_resampling(vector)
        formation = ImageFormationLayer(x_new)
        image = formation.get_reconstructed_image()
        # change RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_path = './DATASET/images/bootstrapping/image_{:06}.png'.format(n)
        cv2.imwrite(image_path, image)

        # create second image with variation
        x_new = self.vector_resampling(vector)
        formation = ImageFormationLayer(x_new)
        image = formation.get_reconstructed_image()
        # change RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_path = './DATASET/images/bootstrapping/image_{:06}.png'.format(n+1)
        cv2.imwrite(image_path, image)


def main():
    # Number of images to read
    N = 4
    path = "/home/anapt/Documents/MUG/cropped/"

    data_root = pathlib.Path(path)
    all_image_paths = list(data_root.glob('*.png'))
    all_image_paths = [str(path) for path in all_image_paths]
    all_image_paths.sort()
    print(all_image_paths)
    all_image_paths = all_image_paths[0:8]

    preprocess = PrepareImages()
    #
    for n, path in enumerate(all_image_paths):
        start = time.time()
        print(path)
        vector = preprocess.get_prediction(path)
        preprocess.create_image_and_save(vector, 2*n)
        print("Time passed:", time.time() - start)
        print(n)


main()