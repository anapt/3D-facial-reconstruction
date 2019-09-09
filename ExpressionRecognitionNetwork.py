from FaceNet3D import FaceNet3D as Helpers
import cv2
import numpy as np
import pathlib
from FaceCropper import FaceCropper
from LandmarkDetection import LandmarkDetection
from InverseFaceNetEncoderPredict import InverseFaceNetEncoderPredict
from ImageFormationLayer import ImageFormationLayer
import time
from LoadDataset import LoadDataset
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

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[self.expression_dim, ]),
            tf.keras.layers.Dense(32, activation=tf.nn.relu),
            tf.keras.layers.Dense(len(self.em), activation=tf.nn.softmax)
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model


def main():
    net = ExpressionRecognition()
    model = net.build_model()

    LoadDataset().load_data_for_expression()

    # model.fit(train_images, train_labels, epochs=5)


main()
