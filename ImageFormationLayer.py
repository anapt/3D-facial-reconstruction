import numpy as np
import matplotlib.pyplot as plt
from SemanticCodeVector import SemanticCodeVector
from ParametricMoDecoder import ParametricMoDecoder
from LandmarkDetection import LandmarkDetection
from ImagePreprocess import ImagePreprocess
from FaceNet3D import FaceNet3D as Helpers


class ImageFormationLayer(Helpers):

    def __init__(self, vector):
        """
        Class initializer

        :param vector: <class 'numpy.ndarray'> with shape (257, ) : semantic code vector
        """
        super().__init__()
        self.vector = self.vector2dict(vector)
        self.preprocess = ImagePreprocess()

    def get_vertices_and_reflectance(self):
        """
        Wrapper function that returns data

        :return:    vertices :      <class 'numpy.ndarray'> with shape (3, 53149)
                    reflectance:    <class 'numpy.ndarray'> with shape (3, 53149)
                    cells:          <class 'numpy.ndarray'> with shape (3, 105694)
        """
        semantic = SemanticCodeVector()
        vertices = semantic.calculate_3d_vertices(self.vector)
        reflectance = semantic.calculate_color(self.vector)

        # read average face cells
        cells = semantic.read_cells()

        return vertices, reflectance, cells

    def get_reconstructed_image(self):
        """
        Wrapper function that returns the reconstructed face (entire image)

        :return: <class 'numpy.ndarray'> with shape (240, 240, 3)
        """
        vertices, reflectance, cells = self.get_vertices_and_reflectance()
        decoder = ParametricMoDecoder(vertices, reflectance, self.vector, cells)

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
        cutout_face = LandmarkDetection().cutout_mask_array(np.uint8(image), flip_rgb=False)

        return cutout_face

    def get_reconstructed_image_for_loss(self):
        """
        Wrapper function that returns the reconstructed face (entire image)

        :return:    cutout_face:    <class 'numpy.ndarray'> with shape (240, 240, 3)
                    indices:        <class 'numpy.ndarray'> with shape (13000,)
                    position:       <class 'numpy.ndarray'> with shape (2, 53149)
        """
        vertices, reflectance, cells = self.get_vertices_and_reflectance()
        decoder = ParametricMoDecoder(vertices, reflectance, self.vector, cells)

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
        cutout_face = LandmarkDetection().cutout_mask_array(np.uint8(image), flip_rgb=False)

        return cutout_face, indices, position


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
