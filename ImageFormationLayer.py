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

        :param vector: <class 'numpy.ndarray'> with shape (self.scv_length, ) : semantic code vector
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
        Wrapper function that returns the reconstructed face (after landmark detection and crop)

        :return: <class 'numpy.ndarray'> with shape self.IMG_SHAPE
        """
        vertices, reflectance, cells = self.get_vertices_and_reflectance()
        decoder = ParametricMoDecoder(vertices, reflectance, self.vector, cells)

        formation = decoder.get_image_formation()
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

    def get_reconstructed_image_no_crop(self):
        """
        Wrapper function that returns the reconstructed face (after landmark detection and crop)

        :return: <class 'numpy.ndarray'> with shape self.IMG_SHAPE
        """
        vertices, reflectance, cells = self.get_vertices_and_reflectance()
        decoder = ParametricMoDecoder(vertices, reflectance, self.vector, cells)

        formation = decoder.get_image_formation()
        cells = decoder.calculate_cell_depth()

        position = formation['position']
        color = formation['color']

        # draw image
        # start = time.time()
        image = self.preprocess.patch(position, color, cells)
        # print("time for patch : ", time.time() - start)

        return image

    def get_reconstructed_image_for_loss(self):
        """
        Wrapper function that returns the reconstructed face in a form that can be used in the loss function

        :return:    cutout_face:    <class 'numpy.ndarray'> with shape self.IMG_SHAPE
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
