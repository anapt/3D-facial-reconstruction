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


class ExpressionRecognition(Helpers):

    def __init__(self):
        """
        Class initializer
        """
        super().__init__()

    def expression_limits(self, _case):
        if _case == 'neutral':
            base = np.zeros((64,))
            limit_left = base - 0.3
            limit_right = base + 0.3

            vector = np.zeros((self.scv_length,))
            vector = self.vector2dict(vector)
            vector['expression'] = base
            np.savetxt("./DATASET/expression/neutral/ground_truth/neutral.txt", vector['expression'])
            # formation = ImageFormationLayer(vector)
            # image = formation.get_reconstructed_image()
            # # change RGB to BGR
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # cv2.imwrite("./DATASET/expression/neutral/ground_truth/neutral.png", image)

            vector = np.zeros((self.scv_length,))
            vector = self.vector2dict(vector)
            vector['expression'] = limit_left
            np.savetxt("./DATASET/expression/neutral/ground_truth/left_limit.txt", vector['expression'])
            # formation = ImageFormationLayer(vector)
            # image = formation.get_reconstructed_image()
            # # change RGB to BGR
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # cv2.imwrite("./DATASET/expression/neutral/ground_truth/left_limit.png", image)

            vector = np.zeros((self.scv_length,))
            vector = self.vector2dict(vector)
            vector['expression'] = limit_right
            np.savetxt("./DATASET/expression/neutral/ground_truth/right_limit.txt", vector['expression'])
            # formation = ImageFormationLayer(vector)
            # image = formation.get_reconstructed_image()
            # # change RGB to BGR
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # cv2.imwrite("./DATASET/expression/neutral/ground_truth/right_limit.png", image)

        if _case == 'happiness':
            limit_left = np.loadtxt("./DATASET/expression/happiness/ground_truth/left_limit.txt")

            limit_right = np.loadtxt("./DATASET/expression/happiness/ground_truth/right_limit.txt")

            base = (limit_left + limit_right) / 2

            print(np.linalg.norm(base - limit_left))
            print(np.linalg.norm(limit_right - base))
            vector = np.zeros((self.scv_length,))
            vector = self.vector2dict(vector)
            vector['expression'] = base
            np.savetxt("./DATASET/expression/happiness/ground_truth/happy.txt", vector['expression'])
            formation = ImageFormationLayer(vector)
            image = formation.get_reconstructed_image()
            # change RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite("./DATASET/expression/happiness/ground_truth/happy.png", image)

            vector = np.zeros((self.scv_length,))
            vector = self.vector2dict(vector)
            vector['expression'] = limit_left
            # np.savetxt("./DATASET/expression/happiness/ground_truth/left_limit.txt", vector['expression'])
            formation = ImageFormationLayer(vector)
            image = formation.get_reconstructed_image()
            # change RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite("./DATASET/expression/happiness/ground_truth/left_limit.png", image)

            vector = np.zeros((self.scv_length,))
            vector = self.vector2dict(vector)
            vector['expression'] = limit_right
            # np.savetxt("./DATASET/expression/happiness/ground_truth/right_limit.txt", vector['expression'])
            formation = ImageFormationLayer(vector)
            image = formation.get_reconstructed_image()
            # change RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite("./DATASET/expression/happiness/ground_truth/right_limit.png", image)

        if _case == 'sadness':
            limit_left = np.loadtxt("./DATASET/expression/sadness/ground_truth/left_limit.txt")

            limit_right = np.loadtxt("./DATASET/expression/sadness/ground_truth/right_limit.txt")
            limit_right = limit_right - 0.02
            base = (limit_left + limit_right) / 2

            print(np.linalg.norm(base - limit_left))
            print(np.linalg.norm(limit_right - base))
            vector = np.zeros((self.scv_length,))
            vector = self.vector2dict(vector)
            vector['expression'] = base
            np.savetxt("./DATASET/expression/sadness/ground_truth/sad.txt", vector['expression'])
            formation = ImageFormationLayer(vector)
            image = formation.get_reconstructed_image()
            # change RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite("./DATASET/expression/sadness/ground_truth/sad.png", image)

            vector = np.zeros((self.scv_length,))
            vector = self.vector2dict(vector)
            vector['expression'] = limit_left
            np.savetxt("./DATASET/expression/sadness/ground_truth/left_limit.txt", vector['expression'])
            formation = ImageFormationLayer(vector)
            image = formation.get_reconstructed_image()
            # change RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite("./DATASET/expression/sadness/ground_truth/left_limit.png", image)

            vector = np.zeros((self.scv_length,))
            vector = self.vector2dict(vector)
            vector['expression'] = limit_right
            np.savetxt("./DATASET/expression/sadness/ground_truth/right_limit.txt", vector['expression'])
            formation = ImageFormationLayer(vector)
            image = formation.get_reconstructed_image()
            # change RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite("./DATASET/expression/sadness/ground_truth/right_limit.png", image)

        if _case == 'surprise':
            limit_left = np.loadtxt("./DATASET/expression/surprise/ground_truth/left_limit.txt")
            limit_left = limit_left - 0.2
            limit_right = np.loadtxt("./DATASET/expression/surprise/ground_truth/right_limit.txt")
            # limit_right = limit_right + 0.1
            base = (limit_left + limit_right) / 2

            print(np.linalg.norm(base - limit_left))
            print(np.linalg.norm(limit_right - base))
            vector = np.zeros((self.scv_length,))
            vector = self.vector2dict(vector)
            vector['expression'] = base
            np.savetxt("./DATASET/expression/surprise/ground_truth/surprised.txt", vector['expression'])
            formation = ImageFormationLayer(vector)
            image = formation.get_reconstructed_image()
            # change RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite("./DATASET/expression/surprise/ground_truth/surprised.png", image)

            vector = np.zeros((self.scv_length,))
            vector = self.vector2dict(vector)
            vector['expression'] = limit_left
            np.savetxt("./DATASET/expression/surprise/ground_truth/left_limit.txt", vector['expression'])
            formation = ImageFormationLayer(vector)
            image = formation.get_reconstructed_image()
            # change RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite("./DATASET/expression/surprise/ground_truth/left_limit.png", image)

            vector = np.zeros((self.scv_length,))
            vector = self.vector2dict(vector)
            vector['expression'] = limit_right
            np.savetxt("./DATASET/expression/surprise/ground_truth/right_limit.txt", vector['expression'])
            formation = ImageFormationLayer(vector)
            image = formation.get_reconstructed_image()
            # change RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            cv2.imwrite("./DATASET/expression/surprise/ground_truth/right_limit.png", image)


def main():
    # happy = np.loadtxt("./DATASET/expression/happiness/ground_truth/left_limit.txt")
    #
    # np.savetxt("./DATASET/expression/sadness/ground_truth/left_limit.txt", -happy)
    #
    # happy = np.loadtxt("./DATASET/expression/happiness/ground_truth/right_limit.txt")
    #
    # np.savetxt("./DATASET/expression/sadness/ground_truth/right_limit.txt", -happy)
    exp = ExpressionRecognition()
    exp.expression_limits(_case='sadness')

main()