import ImagePreprocess as image_preprocess
import time
from FaceNet3D import FaceNet3D as Helpers
from LandmarkDetection import LandmarkDetection
from ParametricMoDecoder import ParametricMoDecoder
from SemanticCodeVector import SemanticCodeVector
import numpy as np
import cv2


def main():
    # Number of images to create
    N = 120
    preprocess = image_preprocess.ImagePreprocess()
    preprocess.create_image_and_save(0)



main()