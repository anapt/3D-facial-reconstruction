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

path = './DATASET/images/validation/image_{:06}.png'.format(35)
img = cv2.imread(path, 1)

# img = FaceCropper().generate(img, save_image=False, n=None)
# cv2.imshow("", img)
# cv2.waitKey(0)
# if img is None:
#     continue
# img = LandmarkDetection().cutout_mask_array(img, flip_rgb=False)
# cv2.imshow("", img)
# cv2.waitKey(0)

# cv2.imwrite("./DATASET/test_cropped.png", img)
# if img is None:
#     continue
# if img.shape != self.IMG_SHAPE:
#     continue
# if fix_color:
#     img = self.fix_color(img)
#
# cv2.imwrite(self.path_mild_images.format(i), img)
# net = InverseFaceNetEncoderPredict()
# vector = net.model_predict(path)
vector = np.loadtxt('./DATASET/semantic/validation/x_{:06}.txt'.format(35))
expression = np.loadtxt('./DATASET/semantic/training/x_{:06}.txt'.format(65))
expression = Helpers().vector2dict(expression)
expression = expression['expression']
np.savetxt('./DATASET/expression/happiness/ground_truth/center.txt', expression*2)
# formation = ImageFormationLayer(vector)
# image = formation.get_reconstructed_image()
# # change RGB to BGR
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# cv2.imwrite("./DATASET/output.png", image)
# cv2.imshow("", image)
# cv2.waitKey(0)

x = Helpers().vector2dict(vector)
x['rotation'] = np.zeros((3,))

emotions = {
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "happiness": 3,
            "neutral": 4,
            "sadness": 5,
            "surprise": 6
        }

for emotion in emotions:
    if emotion == 'happiness':
        path = './DATASET/expression/{}/ground_truth/center.txt'.format(emotion)

        # expression = np.loadtxt(path)
        x['expression'] = expression*2

        formation = ImageFormationLayer(x)
        image = formation.get_reconstructed_image()
        # change RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite("./DATASET/happiness.png".format(emotion), image)
