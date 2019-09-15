from ImageFormationLayer import ImageFormationLayer
from FaceNet3D import FaceNet3D as Helpers
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pathlib

im_path = './DATASET/images/training/image_{:06}.png'
vector_path = './DATASET/semantic/training/x_{:06}.txt'
img = cv2.imread(im_path.format(13))

num = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75]

for i, n in enumerate(num):

    vector = np.loadtxt(vector_path.format(13))
    x = Helpers().vector2dict(vector)
    x['expression'] = x['expression'] * n
    print(np.mean(x['expression']))
    formation = ImageFormationLayer(x)
    image = formation.get_reconstructed_image()
    cv2.imwrite("./DATASET/morphing/{}.png".format(i), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# vector = np.loadtxt(vector_path.format(3))
# x = Helpers().vector2dict(vector)
# x['expression'] = np.convolve(x['expression'], [0.5, 1.5, 0.5])[:64]
# formation = ImageFormationLayer(x)
# image = formation.get_reconstructed_image()
# plt.imshow(image)
# plt.show()

# for i, n in enumerate(num):

    # vector = np.loadtxt('./DATASET/expression/anger/ground_truth/center.txt')
    # print(np.mean(vector*n))
