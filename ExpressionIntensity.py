from FaceNet3D import FaceNet3D as Helpers
import cv2
import numpy as np
import pathlib
from FaceCropper import FaceCropper
from LandmarkDetection import LandmarkDetection
from InverseFaceNetEncoderPredict import InverseFaceNetEncoderPredict
from ImageFormationLayer import ImageFormationLayer
from ExpressionRecognitionNetwork import ExpressionRecognitionNetwork
import time
import tensorflow as tf
import os


class ExpressionIntensity(Helpers):

    def __init__(self, vector):
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

    def read_limits(self):
        expression_limits = {}
        for emotion in self.emotions:
            data_root = './DATASET/expression/{}/ground_truth/'.format(emotion)
            expression = np.loadtxt(data_root + 'center.txt')
            expression = expression * 3

            expression_limits.update({emotion: expression})
        return expression_limits

    def get_prediction(self):
        x = self.network.model_predict_vector(self.vector)
        # print("Expression classified as {}, with confidence {:0.2f}%".format(self.em[int(np.argmax(x))],
        #                                                                      np.amax(x*100)))
        return x

    def get_intensity(self):
        x = self.get_prediction()
        intensity = self.k * (np.mean(self.vector)/np.mean(self.expression_limits[self.em[int(np.argmax(x))]]))
        return intensity

    def get_all(self):
        x = self.get_prediction()
        intensity = self.get_intensity()
        print("Expression classified as {}, with confidence {:0.2f}% and calculated intesity of {:0.2f}/5".
              format(self.em[int(np.argmax(x))], np.amax(x * 100), intensity))
        return None


def main():
    data_root = './DATASET/expression/anger/ground_truth/center.txt'
    vector = np.loadtxt(data_root)
    exp = ExpressionIntensity(vector)
    # print(np.mean(exp.expression_limits['anger']))
    # print(exp.read_limits())
    exp.get_all()

main()
