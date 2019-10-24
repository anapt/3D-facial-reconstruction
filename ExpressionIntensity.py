from FaceNet3D import FaceNet3D as Helpers
import numpy as np
from ExpressionRecognitionNetwork import ExpressionRecognitionNetwork
import pandas as pd


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
        expression_limits = expression_limits*2.5
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
