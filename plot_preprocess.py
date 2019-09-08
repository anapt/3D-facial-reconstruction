import cv2
import numpy as np
import pathlib
from FaceCropper import FaceCropper
from LandmarkDetection import LandmarkDetection


path = "/home/anapt/PycharmProjects/thesis/DATASET/preprocess/image.jpg"
img = cv2.imread(path, 1)

img = FaceCropper().generate(img, save_image=False, n=0)
# cropped_image_path = ("./DATASET/preprocess/{:06}.png".format(0))
# cv2.imwrite(cropped_image_path, img)

img = LandmarkDetection().cutout_mask_array(img, flip_rgb=False)
# img = cv2.resize(img, (224, 224))
# cropped_image_path = ("./DATASET/preprocess/{:06}.png".format(10))
# cv2.imwrite(cropped_image_path, img)