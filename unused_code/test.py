from FaceCropper import FaceCropper
from LandmarkDetection import LandmarkDetection
from InverseFaceNetEncoderPredict import InverseFaceNetEncoderPredict
from FaceNet3D import FaceNet3D as Helpers
import tensorflow as tf
import pandas as pd
from ExpressionRecognitionNetwork import ExpressionRecognitionNetwork
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import pathlib
import numpy as np
import cv2


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


def fix_color(source_color):
    """
    Calculate distance between source_color average color and mean_average_color
    Move source histogram by that distance.
    :param source_color: Original image array (224, 224, 3)
    :return: <class 'numpy.ndarray'> with shape (224, 224, 3)
    """
    mean_average_color = read_average_color()
    color = np.array([0, 0, 0])
    counter = 0
    for i in range(0, source_color.shape[0]):
        for j in range(0, source_color.shape[1]):
            if np.mean(source_color[i, j, :]) > 10:
                color = color + source_color[i, j, :]
                counter = counter + 1

    mean_source_color = color / counter

    constant = (mean_average_color - mean_source_color)*0.5
    print(constant)
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


def prepare_images():
    IMG_SIZE = 224
    COLOR_CHANNELS = 3
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, COLOR_CHANNELS)

    path_wild_images = "/home/anapt/Documents/expression_intensity/"
    path_mild_images = "/home/anapt/Documents/expression_intensity/ready_{:06}.png"
    data_root = pathlib.Path(path_wild_images)

    all_image_paths = list(data_root.glob('*.jpg'))
    all_image_paths = [str(path) for path in all_image_paths]
    all_image_paths.sort()
    # all_image_paths = np.random.choice(all_image_paths, 10, replace=False)

    for i, path in enumerate(all_image_paths):
        img = cv2.imread(path, 1)

        img = FaceCropper().generate(img, save_image=False, n=None)

        if img is None:
            continue
        img = LandmarkDetection().cutout_mask_array(img, flip_rgb=False)
        if img is None:
            continue
        if img.shape != IMG_SHAPE:
            continue
        img = fix_color(img)

        cv2.imwrite(path_mild_images.format(i), img)


prepare_images()


def get_reconstructions():
    path = '/home/anapt/Documents/expression_intensity/'
    data_root = pathlib.Path(path)

    all_image_paths = list(data_root.glob('ready*.png'))
    all_image_paths = [str(path) for path in all_image_paths]
    all_image_paths.sort()
    # all_image_paths = all_image_paths[0:10]

    net = InverseFaceNetEncoderPredict()

    for n, path in enumerate(all_image_paths):
        x = net.model_predict(path)
        np.savetxt("/home/anapt/Documents/expression_intensity/x_{:06}.txt".format(n), x)
        image = net.calculate_decoder_output(x)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite("/home/anapt/Documents/expression_intensity/prediction_{:06}.png".format(n), image)


get_reconstructions()
