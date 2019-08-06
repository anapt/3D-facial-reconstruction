import numpy as np
import ImageFormationLayer as ifl
import matplotlib.pyplot as plt
from InverseFaceNetEncoderPredict import InverseFaceNetPredict
import cv2

# net = InverseFaceNetPredict()

image_path = './landmark_cut.png'

# x = net.model_predict(image_path)
# np.savetxt("./x_boot.txt", x)
x = np.loadtxt("./x_boot.txt")

show_result = True
formation = ifl.ImageFormationLayer(x)
image = formation.get_reconstructed_image()
print(image.shape)

if show_result:
    plt.imshow(image)
    plt.show()

vector = formation.vector


def vector_resampling(vector):
    shape = vector['shape'] + np.random.normal(0, 0.005, 80)

    expression = vector['expression'] + np.random.normal(0, 1, 64)

    reflectance = vector['reflectance'] + np.random.normal(0, 0.2, 80)

    rotation = vector['rotation'] + np.random.uniform(-5, 5, 3)

    translation = np.random.uniform(-0.2, 0.2, 3)
    translation[2] = np.random.normal(0, 0.02, 1)
    translation = vector['translation'] + translation

    illumination = vector['illumination'] + np.random.normal(0, 0.02, 27)

    x = {
        "shape": shape,
        "expression": expression,
        "reflectance": reflectance,
        "rotation": rotation,
        "translation": translation,
        "illumination": illumination
    }

    return x


new_x1 = vector_resampling(vector)
formation = ifl.ImageFormationLayer(new_x1)

image = formation.get_reconstructed_image()
# change RGB to BGR
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_path = './DATASET/images/bootstrapping/image_{:06}.png'.format(1)
cv2.imwrite(image_path, image)

new_x2 = vector_resampling(vector)
formation = ifl.ImageFormationLayer(new_x2)

image = formation.get_reconstructed_image()
# change RGB to BGR
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_path = './DATASET/images/bootstrapping/image_{:06}.png'.format(2)
cv2.imwrite(image_path, image)


# if show_result:
#     plt.imshow(image)
#     plt.show()