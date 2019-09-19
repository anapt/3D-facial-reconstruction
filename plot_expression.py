import pathlib
import os
import numpy as np
import cv2
from InverseFaceNetEncoderPredict import InverseFaceNetEncoderPredict
import pandas as pd
import seaborn
from LossLayer import LossLayer
import matplotlib.pyplot as plt
from prediction_plots import prediction_plots
from FaceNet3D import FaceNet3D as Helpers
from ImageFormationLayer import ImageFormationLayer


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


def read_ground_truth():
    ground_truth = pd.DataFrame(dtype=np.float)
    for key in em:
        path = './DATASET/expression/{}/ground_truth/center.txt'.format(key)
        vector = np.loadtxt(path)
        if key == 'anger' or key == 'disgust':
            vector = vector * 1.5
        else:
            vector = vector*1

        ground_truth.insert(emotions[key], key, vector, True)

    return ground_truth


df = pd.read_csv('./DATASET/expression/expression.csv')
# print(df.head())
print(df.to_latex(index=True, float_format="{:0.4f}".format))


def get_base_images():
    data = pd.read_csv('./DATASET/expression/expression.csv')
    vector = np.zeros((231,))
    vector = Helpers().vector2dict(vector)

    for key in em:
        vector['expression'] = data[key].values
        image = ImageFormationLayer(vector).get_reconstructed_image_no_crop()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('./DATASET/expression/base/{}.png'.format(key), image)


# get_base_images()
