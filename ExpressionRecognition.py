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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
TF_FORCE_GPU_ALLOW_GROWTH = True
tf.compat.v1.enable_eager_execution()
print("\n\n\n\nGPU Available:", tf.test.is_gpu_available())
print("\n\n\n\n")


class ExpressionRecognition(Helpers):

    def __init__(self):
        """
        Class initializer
        """
        super().__init__()
        # self.emotions = {
        #     "anger": 0,
        #     "disgust": 1,
        #     "fear": 2,
        #     "happiness": 3,
        #     "neutral": 4,
        #     "sadness": 5,
        #     "surprise": 6
        # }
        self.emotions = {
            "happiness": 0,
            "neutral": 1,
            "sadness": 2,
            "surprise": 3
        }

        self.em = list(self.emotions.keys())
        self.em.sort()

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

    def prepare_images(self, path, key, fix_color=True):
        """
        Randomly select 5k images, detect faces, detect landmarks and crop
        If necessary, fix color.
        :return: cropped image (224, 224, 3)
        """
        data_root = pathlib.Path(path)

        all_image_paths = list(data_root.glob('*.jpg'))
        all_image_paths = [str(path) for path in all_image_paths]
        all_image_paths = np.random.choice(all_image_paths, 100, replace=False)

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

            new_path = "./DATASET/expression/{}/{:06}.png".format(key, i)
            cv2.imwrite(new_path, img)

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

        constant = (self.mean_average_color - mean_source_color)

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

        shape = vector['shape'] + np.random.normal(0, 0.1, self.shape_dim)

        expression = vector['expression']
                     # + np.random.normal(0, 0.1, self.expression_dim)

        color = vector['color'] + np.random.normal(0, 0.5, self.color_dim)

        rotation = vector['rotation'] + np.random.uniform(-0.5, 0.5, 3)

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

    def create_image_and_save(self, vector, n, path):
        """
        Perform resampling, save new code vector, reconstruct 2d image. save image. Repeat 2 times.
        :param vector: code vector (231,)
        :param n: serial number
        :param path: path
        :return:
        """
        image_path = path + "image_{:06}.png"
        print(image_path)
        vector_path = path + "x_{:06}.txt"
        print(vector_path)
        # create first image with variation
        x_new = self.vector_resampling(vector)
        print(vector_path.format(n))
        np.savetxt(vector_path.format(n), self.dict2vector(x_new))
        formation = ImageFormationLayer(x_new)
        image = formation.get_reconstructed_image()
        # change RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(image_path.format(n), image)
        print(image_path.format(n))
        # create second image with variation
        x_new = self.vector_resampling(vector)
        np.savetxt(vector_path.format(n+1), self.dict2vector(x_new))
        formation = ImageFormationLayer(x_new)
        image = formation.get_reconstructed_image()
        # change RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(image_path.format(n+1), image)

    def create_image(self, vector, n, root):
        # shape = np.random.normal(0, 1, self.shape_dim)
        expression = vector['expression'] + np.random.normal(0, 0.1, self.expression_dim)
        np.savetxt(root + "test_{:06}.txt".format(n), expression)

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
    boot = ExpressionRecognition()
    phase_1 = False

    if phase_1:
        for key in boot.em:
            # print(key)
            root = "/home/anapt/expression_recognition/%s/" % key

            boot.prepare_images(path=root, key=key, fix_color=True)
        # boot.data_augmentation()
    # if not phase_1:
    #     gpus = tf.config.experimental.list_physical_devices('GPU')
    #     print(len(gpus), "Physical GPUs")
    #     if gpus:
    #         # Restrict TensorFlow to only use the first GPU
    #         try:
    #             # tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    #             logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #             print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    #         except RuntimeError as e:
    #             # Visible devices must be set before GPUs have been initialized
    #             print(e)
    #
    #     for key in boot.em:
    #         # print(key)
    #         root = "./DATASET/expression/{}/".format(key)
    #         data_root = pathlib.Path(root)
    #
    #         all_image_paths = list(data_root.glob('*.png'))
    #         all_image_paths = [str(path) for path in all_image_paths]
    #         all_image_paths.sort()
    #         print(all_image_paths)
    #         all_image_paths = all_image_paths[0:5]
    #
    #         for n, path in enumerate(all_image_paths):
    #             start = time.time()
    #             # print(path)
    #             vector = boot.get_prediction(path)
    #             boot.create_image_and_save(vector, 2 * n, root)
    #             # print("Time passed:", time.time() - start)
    #             print(n)
    # vector = boot.get_prediction("./DATASET/expression/neutral/{:06}.png".format(0))
    # vector = np.zeros((231,))
    # np.savetxt("./DATASET/expression/neutral/base.txt", vector)
    # formation = ImageFormationLayer(vector)
    # image = formation.get_reconstructed_image()
    # # change RGB to BGR
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imshow("", image)
    # cv2.waitKey(0)
    for key in boot.em:
        print(key)
        root = "./DATASET/expression/{}/".format(key)
        vector = root + "base.txt".format(5)
        vector = np.loadtxt(vector)
        x = boot.vector2dict(vector)
        for i in range(0, 100):
            boot.create_image(n=i, vector=x, root=root)


main()
