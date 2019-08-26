import cv2
import numpy as np
import matplotlib.pyplot as plt
from FaceNet3D import FaceNet3D as Helpers


path = "./image_{:06}.png".format(0)
average_color = cv2.imread(path, 1)
color = np.array([0, 0, 0])
counter = 0
for i in range(0, average_color.shape[0]):
    for j in range(0, average_color.shape[1]):
        if np.mean(average_color[i, j, :]) > 10:
            color = color + average_color[i, j, :]
            counter = counter + 1

average_color = color/counter
print(average_color)

path = "./{:06}.png".format(0)
source_color = cv2.imread(path, 1)
color = np.array([0, 0, 0])
counter = 0
for i in range(0, source_color.shape[0]):
    for j in range(0, source_color.shape[1]):
        if np.mean(source_color[i, j, :]) > 10:
            color = color + source_color[i, j, :]
            counter = counter + 1

source_color = color/counter
print(source_color)

constant = average_color - source_color

path = "./{:06}.png".format(0)
source_color = cv2.imread(path, 1)
color = np.array([0, 0, 0])
counter = 0
for i in range(0, source_color.shape[0]):
    for j in range(0, source_color.shape[1]):
        if np.mean(source_color[i, j, :]) > 10:
            source_color[i, j, :] = source_color[i, j, :] + constant
        if np.mean(0, source_color[i, j, :]) > 245:
            source_color[i, j, :] = 250




# averaging
# kernel = np.ones((3, 3), np.float32)/9
# dst = cv2.filter2D(img, -1, kernel)
# cv2.imshow("", dst)
# cv2.waitKey(0)
#
# plt.subplot(121), plt.imshow(img), plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
# plt.xticks([]), plt.yticks([])
# plt.show()