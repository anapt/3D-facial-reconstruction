import pathlib
import os
import numpy as np
import cv2
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

emotions = {
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "happiness": 3,
            "neutral": 4,
            "sadness": 5,
            "surprise": 6
        }
em = list(emotions.keys())
em.sort()


def move_images():
    root = "/home/anapt/Documents/MUG/subjects3/"
    data_root = pathlib.Path(root)

    for em in emotions:
        print(em)
        all_image_paths = list(data_root.glob('**/{}/**/*.jpg'.format(em)))
        all_image_paths = [str(path) for path in all_image_paths]
        all_image_paths = np.random.choice(all_image_paths, 2000)

        for i, path in enumerate(all_image_paths):
            new_name = "im_{:06}.jpg".format(i)

            os.system("cp " + path + " /home/anapt/Documents/expression_validation/{}/".format(em) + new_name)


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

    for em in emotions:
        print(em)
        path_wild_images = "/home/anapt/Documents/expression_validation/{}/".format(em)
        path_mild_images = "/home/anapt/Documents/expression_validation/pngs/{}/img_{:06}.png"
        data_root = pathlib.Path(path_wild_images)

        all_image_paths = list(data_root.glob('*.jpg'))
        all_image_paths = [str(path) for path in all_image_paths]
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

            cv2.imwrite(path_mild_images.format(em, i), img)


encoder = InverseFaceNetEncoderPredict()


def get_prediction(image_path):
    """
    Use trained model to predict code vector
    :param image_path: path to image
    :return: code vector
    """
    vector = encoder.model_predict(image_path=image_path)
    vector = Helpers().vector2dict(vector)
    expression = vector['expression']
    x = ExpressionRecognitionNetwork().model_predict_vector(expression)
    # print("Expression classified as {}, with confidence {:0.2f}%".format(em[int(np.argmax(x))],
    #                                                                      np.amax(x*100)))

    return x


d = {'true_label': [], 'predicted_label': []}
df = pd.DataFrame(data=d, dtype=np.int64)
for em in emotions:
    path = "/home/anapt/Documents/expression_validation/clean/{}/".format(em)
    data_root = pathlib.Path(path)

    all_image_paths = list(data_root.glob('*.png'))
    all_image_paths = [str(path) for path in all_image_paths]
    # all_image_paths = all_image_paths[0:10]
    all_image_paths.sort()
    print(em)
    for i, path in enumerate(all_image_paths):
        x = get_prediction(path)
        # print(x)

        df = df.append({'true_label': emotions[em], 'predicted_label': np.argmax(x)}, ignore_index=True)

export_csv = df.to_csv(r'/home/anapt/export_dataframe.csv', index=None, header=True)


def get_confusion_matrix():
    data = pd.read_csv("/home/anapt/export_dataframe.csv")

    confusion_mat = confusion_matrix(data['true_label'], data['predicted_label'])

    labels = ["ANGRY", "DISGUST", "FEAR", "HAPPY", "NEUTRAL", "SAD", "SURPRISE"]

    plt.figure(figsize=(16, 9))
    seaborn.heatmap(confusion_mat, cmap="Blues", annot=True, fmt=".1f", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix", fontsize=20)
    plt.xlabel('Predicted Class', fontsize=10)
    plt.ylabel('Original Class', fontsize=10)
    plt.tick_params(labelsize=7)
    plt.xticks(rotation=0)
    plt.yticks(rotation=90)
    plt.savefig("/home/anapt/conf_boot1.pdf")
