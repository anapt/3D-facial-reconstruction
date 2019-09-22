from FaceNet3D import FaceNet3D as Helpers
import cv2
import numpy as np
import pathlib
from FaceCropper import FaceCropper
from LandmarkDetection import LandmarkDetection
from InverseFaceNetEncoderPredict import InverseFaceNetEncoderPredict
from ImageFormationLayer import ImageFormationLayer
from ExpressionRecognitionNetwork import ExpressionRecognitionNetwork
import pandas as pd
import time
import tensorflow as tf
import os


class ExpressionIntensity(Helpers):

    def __init__(self, vector=0):
        """
        Class initializer
        """
        super().__init__()
        self.emotions = {
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "happiness": 3,
            "neutral": 4,
            "sadness": 5,
            "surprise": 6
        }

        self.em = list(self.emotions.keys())
        self.em.sort()
        self.expression_limits = self.read_limits()
        self.vector = vector
        self.network = ExpressionRecognitionNetwork()
        self.network.load_model()
        self.k = 5
        # self.encoder = InverseFaceNetEncoderPredict()

    @staticmethod
    def read_limits():
        expression_limits = pd.read_csv('./DATASET/expression/expression.csv')
        # print(expression_limits.head())
        x = np.loadtxt("/home/anapt/Documents/expression_intensity/x_{:06}.txt".format(5))
        x = Helpers().vector2dict(x)
        expression_limits['sadness'] = x['expression']
        expression_limits = expression_limits*1.2
        return expression_limits

    def get_prediction(self):
        x = self.network.model_predict_vector(self.vector)
        # print("Expression classified as {}, with confidence {:0.2f}%".format(self.em[int(np.argmax(x))],
        #                                                                      np.amax(x*100)))
        return x

    def get_intensity(self):
        x = self.get_prediction()
        intensity = self.k * abs(np.mean(self.vector)/np.mean(self.expression_limits[self.em[int(np.argmax(x))]]))
        print(np.mean(self.vector))
        return intensity

    def get_all(self):
        x = self.get_prediction()
        intensity = self.get_intensity()
        print("Expression classified as {}, with confidence {:0.2f}% and calculated intesity of {:0.2f}/5".
              format(self.em[int(np.argmax(x))], np.amax(x * 100), intensity))
        return self.em[int(np.argmax(x))], np.amax(x * 100), intensity

    # def get_encoding(self, image_path):
    #
    #     vector = self.encoder.model_predict(image_path=image_path)
    #     vector = Helpers().vector2dict(vector)
    #     expression = vector['expression']
    #
    #     return expression


def main():
    path = '/home/anapt/Documents/expression_intensity/'
    data_root = pathlib.Path(path)

    all_vector_paths = list(data_root.glob('x*.txt'))
    all_vector_paths = [str(path) for path in all_vector_paths]
    all_vector_paths.sort()

    d = {'prediction': [], 'confidence': [], 'intensity': []}
    df = pd.DataFrame(data=d, dtype=np.float)
    for n, path in enumerate(all_vector_paths):
        x = np.loadtxt(path)
        x = Helpers().vector2dict(x)
        expression = x['expression']

        exp = ExpressionIntensity(expression)
        # print(np.mean(exp.expression_limits['anger']))
        # print(exp.read_limits())
        pred, conf, inten = exp.get_all()

        df = df.append({'prediction': pred, 'confidence': conf, 'intensity': inten}, ignore_index=True)

    export_csv = df.to_csv(r'./intensity1.csv', index=None, header=True)


main()
