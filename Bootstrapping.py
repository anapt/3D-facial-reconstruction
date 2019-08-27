from FaceNet3D import FaceNet3D as Helpers
import cv2
import numpy as np
import pathlib
from FaceCropper import FaceCropper
from LandmarkDetection import LandmarkDetection
from InverseFaceNetEncoderPredict import InverseFaceNetEncoderPredict
from ImageFormationLayer import ImageFormationLayer
import time


class Bootstrapping(Helpers):

    def __init__(self):
        """
        Class initializer
        """
        super().__init__()
        # path with wild images
        self.path_wild_images = "/home/anapt/Documents/MUG/unpacked/"
        # path to store images after pre-process (crop and color fix)
        self.path_mild_images = "./DATASET/bootstrapping/MUG/{:06}.png"
        # paths for bootstrapping training set
        self.image_path = "./DATASET/bootstrapping/images/image_{:06}.png"
        self.vector_path = "./DATASET/bootstrapping/semantic/x_{:06}.txt"

        self.mean_average_color = self.read_average_color()
        self.net = InverseFaceNetEncoderPredict()

    def prepare_images(self):
        data_root = pathlib.Path(self.path_wild_images)

        all_image_paths = list(data_root.glob('*.jpg'))
        all_image_paths = [str(path) for path in all_image_paths]
        all_image_paths = np.random.choice(all_image_paths, 5100)
        print(all_image_paths)
        for i, path in enumerate(all_image_paths):
            img = cv2.imread(path, 1)

            img = FaceCropper().generate(img, save_image=False, n=None)
            if img is None:
                continue
            img = LandmarkDetection().cutout_mask_array(img, flip_rgb=False)
            if img.shape != self.IMG_SHAPE:
                continue

            img = self.fix_color(img)

            cv2.imwrite(self.path_mild_images.format(i), img)

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

        constant = self.mean_average_color - mean_source_color

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

        shape = vector['shape'] + np.random.normal(0, 0.05, self.shape_dim)

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
        x_new = self.vector_resampling(vector)
        np.savetxt(self.vector_path.format(n), self.dict2vector(x_new))
        formation = ImageFormationLayer(x_new)
        image = formation.get_reconstructed_image()
        # change RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(self.image_path.format(n), image)

        # create second image with variation
        x_new = self.vector_resampling(vector)
        np.savetxt(self.vector_path.format(n+1), self.dict2vector(x_new))

        formation = ImageFormationLayer(x_new)
        image = formation.get_reconstructed_image()
        # change RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(self.image_path.format(n+1), image)


def main():
    boot = Bootstrapping()
    phase_1 = True

    if phase_1:
        boot.prepare_images()

    if not phase_1:
        path = "./DATASET/bootstrapping/MUG/"
        data_root = pathlib.Path(path)
        all_image_paths = list(data_root.glob('*.png'))
        all_image_paths = [str(path) for path in all_image_paths]
        all_image_paths.sort()
        print(all_image_paths)
        # all_image_paths = all_image_paths[0:5000]

        for n, path in enumerate(all_image_paths):
            start = time.time()
            print(path)
            vector = boot.get_prediction(path)
            boot.create_image_and_save(vector, 2 * n)
            # print("Time passed:", time.time() - start)
            print(n)

main()
