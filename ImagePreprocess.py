from FaceNet3D import FaceNet3D as Helpers
from LandmarkDetection import LandmarkDetection
from ParametricMoDecoder import ParametricMoDecoder
from SemanticCodeVector import SemanticCodeVector
import numpy as np
import cv2


class ImagePreprocess(Helpers):

    def __init__(self):
        super().__init__()
        self.cut = LandmarkDetection()
        self.data = SemanticCodeVector()

    def get_vectors(self, n):
        """
        Samples vector, saves vector in .txt file
        Calculate image formation (2d coordinates and color)

        :param n: iteration number
        :return:    image formation (dictionary with keys position, color)
                    cell ordered with deepest one first
        """

        cells = self.data.read_cells()

        # x = data.sample_vector()
        x = self.data.sample_vector()

        vector = self.dict2vector(x)

        np.savetxt(self.vector_path.format(n), vector)

        vertices = self.data.calculate_3d_vertices(x)
        color = self.data.calculate_color(x)

        decoder = ParametricMoDecoder(vertices, color, x, cells)

        formation = decoder.get_image_formation()

        cells_ordered = decoder.calculate_cell_depth()

        return formation, cells_ordered

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

        # cv2.imshow("", rotated180)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

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

        if self.testing:
            rotated180 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(self.no_crop_path.format(n), rotated180)
            image = cv2.cvtColor(rotated180, cv2.COLOR_BGR2RGB)

        # get face mask without mouth interior
        cut = LandmarkDetection()
        # RGB image with face
        out_face = cut.cutout_mask_array(np.uint8(image), True)

        cropped_image_path = (self.cropped_path.format(n))
        cv2.imwrite(cropped_image_path, out_face)
