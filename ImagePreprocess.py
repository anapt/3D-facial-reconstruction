import LandmarkDetection as ld
from unused import FaceCropper as fc
import ParametricMoDecoder as pmd
import SemanticCodeVector as scv
import time
import numpy as np
import cv2


class ImagePreprocess(object):

    def __init__(self):
        self.cut = ld.LandmarkDetection()
        self.crop = fc.FaceCropper()
        self.data = scv.SemanticCodeVector('./DATASET/model2017-1_bfm_nomouth.h5')
        self.path = './DATASET/model2017-1_bfm_nomouth.h5'

    def get_vectors(self, n):
        """
        Samples vector, saves vector in .txt file
        Calculate image formation (2d coordinates and color)

        :param n: iteration number
        :return:    image formation (dictionary with keys position, color)
                    cell ordered with deepest one first
        """
        data = scv.SemanticCodeVector(self.path)
        cells = data.read_cells()

        x = data.sample_vector()

        vector = np.zeros(257, dtype=float)
        vector[0:80, ] = x['shape']
        vector[80:144, ] = x['expression']
        vector[144:224, ] = x['reflectance']
        vector[224:227, ] = x['rotation']
        vector[227:230, ] = x['translation']
        vector[230:257, ] = x['illumination']

        np.savetxt("./DATASET/semantic/x_{:06}.txt".format(n), vector)

        vertices = data.calculate_coords(x)
        reflectance = data.calculate_reflectance(x)

        decoder = pmd.ParametricMoDecoder(vertices, reflectance, x, cells)

        formation = decoder.get_image_formation()

        cells_ordered = decoder.calculate_cell_depth()

        return formation, cells_ordered

    @staticmethod
    def translate(value, left_min, left_max, right_min=0, right_max=500):
        """
        Translates coordinates from range [left_min, left_max]
        to range [right_min, right_max]

        :param value:       value to translate
        :param left_min:    float
        :param left_max:    float
        :param right_min:   float
        :param right_max:   float
        :return: same shape and type as value
        """
        # Figure out how 'wide' each range is
        left_span = left_max - left_min
        right_span = right_max - right_min

        # Convert the left range into a 0-1 range (float)
        # print(np.subtract(value, leftMin))
        value_scaled = np.subtract(value, left_min) / float(left_span)

        # Convert the 0-1 range into a value in the right range.
        # print(right_min + (value_scaled * right_span))
        return right_min + (value_scaled * right_span)

    def patch(self, position, color, cells):
        """
        Drawing function

        :param position:    projected coordinates of the vertices
                            <class 'numpy.ndarray'> with shape (2, 53149)
        :param color:       color of the vertices
                            <class 'numpy.ndarray'> with shape (3, 53149)
        :param cells:       array containing the connections between vertices
                            <class 'numpy.ndarray'> with shape (3, 50000)
        :return:            drawn image
                            <class 'numpy.ndarray'> with shape (500, 500, 3)
        """
        n_cells = cells.shape[1]
        w = 500
        image = np.zeros((w, w, 3), dtype=np.uint8)

        coord = np.zeros(shape=(3, 2, n_cells))

        for i in range(0, n_cells):
            triangle = cells[:, i]
            x = position[0, triangle]
            y = position[1, triangle]
            coord[:, :, i] = np.transpose(np.vstack((x, y)))

        coord = self.translate(coord, np.amin(coord), np.amax(coord), 130, 370)

        for i in range(0, n_cells):
            triangle = cells[:, i]

            tri_color = color[:, triangle]
            triangle_color = (np.average(tri_color, axis=1)) * 255

            cv2.fillConvexPoly(image, np.int64([coord[:, :, i]]), color=tuple([int(x) for x in triangle_color]))

        # rotate drawn image
        center = (w / 2, w / 2)
        angle180 = 180
        scale = 1.0

        rotation_matrix = cv2.getRotationMatrix2D(center, angle180, scale)
        rotated180 = cv2.warpAffine(image, rotation_matrix, (w, w))
        return rotated180

    def create_image_and_save(self, n):
        """
        Wrapper function that calls the preprocess with the correct order

        :param n: number of image (int)
        :return:
        """
        formation, cells = self.get_vectors(n)
        position = formation['position']
        color = formation['color']

        # create image
        image = self.patch(position, color, cells)

        # get face mask without mouth interior
        cut = ld.LandmarkDetection()
        # RGB image with face
        cut.cutout_mask_array(np.uint8(image), n, True, True)
