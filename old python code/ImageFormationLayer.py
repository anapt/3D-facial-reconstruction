import numpy as np
import matplotlib.pyplot as plt

import SemanticCodeVector as scv
import ParametricMoDecoder as pmd
import LandmarkDetection as ld
import ImagePreprocess as preprocess


class ImageFormationLayer(object):

    def __init__(self, vector):
        """
        Class initializer

        :param vector: <class 'numpy.ndarray'> with shape (257, ) : semantic code vector
        """
        self.vector = self.vector2dict(vector)
        self.path = './DATASET/model2017-1_bfm_nomouth.h5'
        self.preprocess = preprocess.ImagePreprocess()

    def get_vertices_and_reflectance(self):
        """
        Wrapper function that returns data

        :return:    vertices :      <class 'numpy.ndarray'> with shape (3, 53149)
                    reflectance:    <class 'numpy.ndarray'> with shape (3, 53149)
                    cells:          <class 'numpy.ndarray'> with shape (3, 105694)
        """
        semantic = scv.SemanticCodeVector(self.path)
        vertices = semantic.calculate_coords(self.vector)
        reflectance = semantic.calculate_reflectance(self.vector)

        # read average face cells
        cells = semantic.read_cells()

        return vertices, reflectance, cells

    def get_reconstructed_image(self):
        """
        Wrapper function that returns the reconstructed face (entire image)

        :return: <class 'numpy.ndarray'> with shape (240, 240, 3)
        """
        vertices, reflectance, cells = self.get_vertices_and_reflectance()
        decoder = pmd.ParametricMoDecoder(vertices, reflectance, self.vector, cells)

        formation = decoder.get_image_formation()
        # get cell depth and keep only 50000 cells
        cells = decoder.calculate_cell_depth()

        position = formation['position']
        color = formation['color']

        # draw image
        # start = time.time()
        image = self.preprocess.patch(position, color, cells)
        # print("time for patch : ", time.time() - start)

        # get face mask without mouth interior
        cut = ld.LandmarkDetection()
        cutout_face = cut.cutout_mask_array(np.uint8(image), n=None, flip_rgb=False, save_image=False)

        return cutout_face

    def get_reconstructed_image_for_loss(self):
        """
        Wrapper function that returns the reconstructed face (entire image)

        :return:    cutout_face:    <class 'numpy.ndarray'> with shape (240, 240, 3)
                    indices:        <class 'numpy.ndarray'> with shape (13000,)
                    position:       <class 'numpy.ndarray'> with shape (2, 53149)
        """
        vertices, reflectance, cells = self.get_vertices_and_reflectance()
        decoder = pmd.ParametricMoDecoder(vertices, reflectance, self.vector, cells)

        formation = decoder.get_image_formation()
        cells = decoder.calculate_cell_depth()

        # sample indices
        indices = np.unique(cells).astype(int)
        indices = np.random.choice(indices, size=13000, replace=False)

        position = formation['position']
        color = formation['color']

        # draw image
        # start = time.time()
        image = self.preprocess.patch(position, color, cells)
        # print("time for patch : ", time.time() - start)

        # get face mask without mouth interior
        cut = ld.LandmarkDetection()
        cutout_face = cut.cutout_mask_array(np.uint8(image), n=None, save_image=False, flip_rgb=False)

        return cutout_face, indices, position

    @staticmethod
    def vector2dict(vector):
        """
        Method that transforms (257,) nd.array to dictionary

        :param vector: <class 'numpy.ndarray'> with shape (257, ) : semantic code vector
        :return:
        dictionary with keys    shape           (80,)
                                expression      (64,)
                                reflectance     (80,)
                                rotation        (3,)
                                translation     (3,)
                                illumination    (27,)
        """
        if isinstance(vector, dict):
            return vector
        else:
            x = {
                "shape": np.squeeze(vector[0:80, ]),
                "expression": np.squeeze(vector[80:144, ]),
                "reflectance": np.squeeze(vector[144:224, ]),
                "rotation": np.squeeze(vector[224:227, ]),
                "translation": np.squeeze(vector[227:230, ]),
                "illumination": np.squeeze(vector[230:257, ])
            }
            return x


def main():
    show_result = False
    n = 0
    vector_path = ("./DATASET/semantic/x_{:06}.txt".format(n))
    vector = np.loadtxt(vector_path)

    formation = ImageFormationLayer(vector)
    image = formation.get_reconstructed_image_for_loss()

    if show_result:
        plt.imshow(image)
        plt.show()


# main()
