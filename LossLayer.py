import numpy as np
import matlab.engine
import matplotlib.pyplot as plt
import ImageFormationLayer as ifl
import semanticCodeVector as scv
import parametricMoDecoder as pmd
import LandmarkDetection as ld
import FaceCropper as fc
import cv2


class LossLayer:
    PATH = './DATASET/model2017-1_bfm_nomouth.h5'

    def __init__(self, vector):
        self.x = {
            "shape": vector[0:80, ],
            "expression": vector[80:144, ],
            "reflectance": vector[144:224, ],
            "rotation": vector[224:227, ],
            "translation": vector[227:230, ],
            "illumination": vector[230:257, ]
            }

    def statistical_regularization_term(self):
        weight_expression = 0.8
        weight_reflectance = 1.7e-3
        sr_term = sum(self.x['shape']) + weight_expression * sum(self.x['expression']) + \
            weight_reflectance * sum(self.x['reflectance'])

        return sr_term

    def dense_photometric_alignment(self, original_image):
        # TODO original_image has to come from the tf.dataset
        formation = ifl.ImageFormationLayer(self.PATH, self.x)
        new_image, position = formation.get_reconstructed_image()
        print(new_image.shape)
        print(position)
        print(position.shape)
        print(type(position))
        photo_term = original_image[position[0], position[1], :] - new_image

        plt.imshow(new_image)
        plt.show()


def main():
    show_result = True
    n = 5
    vector_path = ("./DATASET/semantic/x_%d.txt" % n)
    image_path = ("./DATASET/images/image_%d.png" % n)
    vector = np.loadtxt(vector_path)
    # print(vector.shape)
    # vector = np.ones((257, ))
    # print(vector.shape)
    x = {
        "shape": vector[0:80, ],
        "expression": vector[80:144, ],
        "reflectance": vector[144:224, ],
        "rotation": vector[224:227, ],
        "translation": vector[227:230, ],
        "illumination": vector[230:257, ]
    }

    ll = LossLayer(vector)
    sr_term = ll.statistical_regularization_term()
    print(sr_term)
    original_image = cv2.imread(image_path)
    print(type(original_image))
    ll.dense_photometric_alignment(original_image)


main()