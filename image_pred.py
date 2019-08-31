from FaceNet3D import FaceNet3D as Helpers
import cv2
import numpy as np
import pathlib
from FaceCropper import FaceCropper
from LandmarkDetection import LandmarkDetection
from InverseFaceNetEncoderPredict import InverseFaceNetEncoderPredict
from ImageFormationLayer import ImageFormationLayer
import time
import tensorflow as tf
import os

class Bootstrapping(Helpers):

    def __init__(self):
        """
        Class initializer
        """
        super().__init__()
        # path with wild images
        self.path_wild_images = "/home/anapt/Documents/MUG/unpacked/"
        # self.path_wild_images = "/home/anapt/Documents/dataset2/"
        # path to store images after pre-process (crop and color fix)
        self.path_mild_images = "./DATASET/bootstrapping/MUG/{:06}.png"
        # self.path_mild_images = "./DATASET/bootstrapping/FERET/{:06}.png"
        # paths for bootstrapping training set

        self.image_path = "./DATASET/bootstrapping/images/image_{:06}.png"
        self.vector_path = "./DATASET/bootstrapping/semantic/x_{:06}.txt"

        self.mean_average_color = self.read_average_color()
        self.net = InverseFaceNetEncoderPredict()

    def prepare_images(self):
        data_root = pathlib.Path(self.path_wild_images)

        all_image_paths = list(data_root.glob('*.jpg'))
        all_image_paths = [str(path) for path in all_image_paths]
        all_image_paths = np.random.choice(all_image_paths, 5, replace=False)
        # print(all_image_paths)
        # all_image_paths = all_image_paths[0:10]
        # print("here", all_image_paths)
        for i, path in enumerate(all_image_paths):
            img = cv2.imread(path, 1)
            cv2.imwrite("/home/anapt/predictions/original_image_{:06}.png".format(i), img)
            img = FaceCropper().generate(img, save_image=False, n=None)
            # cv2.imshow("", img)
            # cv2.waitKey(0)
            if img is None:
                continue
            img = LandmarkDetection().cutout_mask_array(img, flip_rgb=False)
            if img is None:
                continue
            if img.shape != self.IMG_SHAPE:
                continue
            cv2.imwrite("/home/anapt/predictions/after_preprocess_{:06}.png".format(i), img)
            img = self.fix_color(img)

            # cv2.imwrite(self.path_mild_images.format(2983+i), img)
            cv2.imwrite("/home/anapt/predictions/after_color_fix_{:06}.png".format(i), img)


    @staticmethod
    def read_average_color():
        average_color = cv2.imread("./DATASET/bootstrapping/average_color.png", 1)
        color = np.array([0, 0, 0])
        counter = 0
        for i in range(0, average_color.shape[0]):
            for j in range(0, average_color.shape[1]):
                if np.mean(average_color[i, j, :]) > 10:
                    color = color + average_color[i, j, :]
                    counter = counter + 1

        mean_average_color = color / counter

        return mean_average_color

    def fix_color(self, source_color):
        color = np.array([0, 0, 0])
        counter = 0
        for i in range(0, source_color.shape[0]):
            for j in range(0, source_color.shape[1]):
                if np.mean(source_color[i, j, :]) > 10:
                    color = color + source_color[i, j, :]
                    counter = counter + 1

        mean_source_color = color / counter

        constant = (self.mean_average_color - mean_source_color)*0.5

        for i in range(0, source_color.shape[0]):
            for j in range(0, source_color.shape[1]):
                if np.mean(source_color[i, j, :]) > 5:
                    if source_color[i, j, 0] + constant[0] > 255:
                        source_color[i, j, 0] = 255
                    else:
                        source_color[i, j, 0] = source_color[i, j, 0] + constant[0]

                    if source_color[i, j, 1] + constant[1] > 255:
                        source_color[i, j, 1] = 255
                    else:
                        source_color[i, j, 1] = source_color[i, j, 1] + constant[1]

                    if source_color[i, j, 2] + constant[2] > 255:
                        source_color[i, j, 2] = 255
                    else:
                        source_color[i, j, 2] = source_color[i, j, 2] + constant[2]

        return source_color

    def vector_resampling(self, vector):
        vector = self.vector2dict(vector)

        shape = vector['shape'] + np.random.normal(0, 0.1, self.shape_dim)

        expression = vector['expression'] + np.random.normal(0, 0.1, self.expression_dim)

        color = vector['color'] + np.random.normal(0, 1, self.color_dim)

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
        return vector

    def create_image_and_save(self, vector, n):
        # create first image with variation

        formation = ImageFormationLayer(vector)
        image = formation.get_reconstructed_image()
        # change RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite("/home/anapt/predictions/prediction_after_color_fix_{:06}.png".format(n), image)



def main():
    boot = Bootstrapping()
    phase_1 = False

    if phase_1:
        boot.prepare_images()

    if not phase_1:
        with tf.device('/device:CPU:0'):
            path = "/home/anapt/predictions/"

            for i in range(0, 5):
                print(path+'after_color_fix_{:06}.png'.format(i))
                if os.path.exists(path+'after_color_fix_{:06}.png'.format(i)):    # True
                    vector = boot.get_prediction(path+'after_color_fix_{:06}.png'.format(i))
                    boot.create_image_and_save(vector, i)
                # print("Time passed:", time.time() - start)
                print(i)

main()