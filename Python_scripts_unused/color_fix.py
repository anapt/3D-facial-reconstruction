import cv2
import numpy as np
import matplotlib.pyplot as plt
from FaceNet3D import FaceNet3D as Helpers
import pathlib

path = "/home/anapt/Documents/MUG/cropped/"

data_root = pathlib.Path(path)

all_image_paths = list(data_root.glob('*.png'))
all_image_paths = [str(path) for path in all_image_paths]
print(len(all_image_paths))
all_image_paths.sort()

average_color = cv2.imread("./average_color.png", 1)
color = np.array([0, 0, 0])
counter = 0
for i in range(0, average_color.shape[0]):
    for j in range(0, average_color.shape[1]):
        if np.mean(average_color[i, j, :]) > 10:
            color = color + average_color[i, j, :]
            counter = counter + 1

mean_average_color = color / counter


for k, path in enumerate(all_image_paths):

    source_color = cv2.imread(path, 1)
    color = np.array([0, 0, 0])
    counter = 0
    for i in range(0, source_color.shape[0]):
        for j in range(0, source_color.shape[1]):
            if np.mean(source_color[i, j, :]) > 10:
                color = color + source_color[i, j, :]
                counter = counter + 1

    mean_source_color = color/counter

    constant = mean_average_color - mean_source_color

    write_path = "./DATASET/images/MUG/{:06}.png".format(k)
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

    cv2.imwrite(write_path, source_color)

img = cv2.imread("./DATASET/images/MUG/{:06}.png".format(1), 1)
cv2.imshow("", img)
cv2.waitKey()