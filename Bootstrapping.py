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
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# # TF_FORCE_GPU_ALLOW_GROWTH = False
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# tf.keras.backend.set_session(sess)
#
# print("\n\n\n\nGPU Available:", tf.test.is_gpu_available())
# print("\n\n\n\n")


class Bootstrapping(Helpers):

    def __init__(self):
        """
        Class initializer
        """
        super().__init__()
        # path with wild images
        # TODO add path to images
        self.path_wild_images = "/home/anapt/Documents/MUG/unpacked/"
        # path to store images after pre-process (crop and color fix)
        self.path_mild_images = "./DATASET/bootstrapping/MUG/{:06}.png"

        # paths for bootstrapping training set
        self.image_path = "./DATASET/bootstrapping/images/image_{:06}.png"
        self.vector_path = "./DATASET/bootstrapping/semantic/x_{:06}.txt"

        self.mean_average_color = self.read_average_color()
        self.net = InverseFaceNetEncoderPredict()

    def prepare_images(self, fix_color=True):
        """
        Randomly select 5k images, detect faces, detect landmarks and crop
        If necessary, fix color.
        :return: cropped image (224, 224, 3)
        """
        data_root = pathlib.Path(self.path_wild_images)

        all_image_paths = list(data_root.glob('*.jpg'))
        all_image_paths = [str(path) for path in all_image_paths]
        all_image_paths = np.random.choice(all_image_paths, 5000, replace=False)

        for i, path in enumerate(all_image_paths):
            img = cv2.imread(path, 1)

            img = FaceCropper().generate(img, save_image=False, n=None)

            if img is None:
                continue
            img = LandmarkDetection().cutout_mask_array(img, flip_rgb=False)
            if img is None:
                continue
            if img.shape != self.IMG_SHAPE:
                continue
            if fix_color:
                img = self.fix_color(img)

            cv2.imwrite(self.path_mild_images.format(i), img)

    @staticmethod
    def read_average_color():
        """
        Read image with "average color" and find that image's histogram mean
        Exclude black parts of the images.
        :return: (3, ) average color
        """
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
        """
        Calculate distance between source_color average color and mean_average_color
        Move source histogram by that distance.
        :param source_color: Original image array (224, 224, 3)
        :return: <class 'numpy.ndarray'> with shape (224, 224, 3)
        """
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
        """
        Re-sample vector for bootstrapping
        This function adds gaussian noise to the vector to better fit real world images
        :param vector: code vector (231,)
        :return: re-sampled code vector (231,)
        """
        vector = self.vector2dict(vector)

        shape = vector['shape'] + np.random.normal(0, 0.2, self.shape_dim)

        expression = vector['expression'] + np.random.normal(0, 0.2, self.expression_dim)

        color = vector['color'] + np.random.normal(0, 0.2, self.color_dim)

        rotation = vector['rotation']

        x = {
            "shape": shape,
            "expression": expression,
            "color": color,
            "rotation": rotation,
        }

        return x

    def get_prediction(self, image_path):
        """
        Use trained model to predict code vector
        :param image_path: path to image
        :return: code vector
        """
        vector = self.net.model_predict(image_path=image_path)
        return vector

    def create_image_and_save(self, vector, n):
        """
        Perform resampling, save new code vector, reconstruct 2d image. save image. Repeat 2 times.
        :param vector: code vector (231,)
        :param n: serial number
        :return:
        """
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

    def data_augmentation(self):
        data_root = pathlib.Path(self.bootstrapping_path + 'MUG/')

        all_image_paths = list(data_root.glob('*.png'))
        all_image_paths = [str(path) for path in all_image_paths]
        # all_image_paths = np.random.choice(all_image_paths, 10, replace=False)

        for i, path in enumerate(all_image_paths):
            img = cv2.imread(path, 1)
            img = cv2.flip(img, 1)

            angle = np.random.uniform(-5, 5, 1)
            M = cv2.getRotationMatrix2D((self.IMG_SIZE / 2, self.IMG_SIZE / 2), angle, 1.0)
            dst = cv2.warpAffine(img, M, (self.IMG_SIZE, self.IMG_SIZE))

            cv2.imwrite(self.path_mild_images.format(i), dst)


def main():
    boot = Bootstrapping()
    phase_1 = True

    if phase_1:
        boot.prepare_images(fix_color=True)
        # boot.data_augmentation()
    if not phase_1:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7 * 1024)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)

        path = "./DATASET/bootstrapping/predict/"
        data_root = pathlib.Path(path)
        all_image_paths = list(data_root.glob('*.png'))
        all_image_paths = [str(path) for path in all_image_paths]
        all_image_paths.sort()
        # print(all_image_paths)
        all_image_paths = all_image_paths[10000:15000]

        for n, path in enumerate(all_image_paths):
            start = time.time()
            # print(path)
            vector = boot.get_prediction(path)
            boot.create_image_and_save(vector, 20000 + 2 * n)
            # print("Time passed:", time.time() - start)
            print(n)


# main()
