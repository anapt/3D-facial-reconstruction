import LandmarkDetection as ld
import FaceCropper as fc
import ParametricMoDecoder as pmd
import SemanticCodeVector as scv

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

        :param path: Path to Basel Face Model
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

        np.savetxt("./DATASET/semantic/x_%d.txt" % n, vector)

        vertices = data.calculate_coords(x)
        reflectance = data.calculate_reflectance(x)

        decoder = pmd.ParametricMoDecoder(vertices, reflectance, x, cells)

        formation = decoder.get_image_formation()
        # np.savetxt("./DATASET/color/color_%d.txt" % n, formation['color'])
        # np.savetxt("./DATASET/position/position_%d.tgxt" % n, formation['position'])

        cells_ordered = decoder.calculate_cell_depth()
        # np.savetxt("DATASET/cells/cells_%d.txt" % n, cells_ordered)

        return formation, cells_ordered

    @staticmethod
    def translate(value, left_min, left_max):
        # Figure out how 'wide' each range is
        right_min = 0
        right_max = 500
        left_span = left_max - left_min
        right_span = right_max - right_min

        # Convert the left range into a 0-1 range (float)
        # print(np.subtract(value, leftMin))
        value_scaled = np.subtract(value, left_min) / float(left_span)

        # Convert the 0-1 range into a value in the right range.
        return right_min + (value_scaled * right_span)

    def patch(self, position, color, cells):
        n_cells = 105694
        w = 500
        image = np.zeros((w, w, 3), dtype=np.uint8)

        for i in range(n_cells - 50000, n_cells):
            triangle = cells[:, i]
            # print(i)
            # print(triangle)
            x = position[0, triangle]
            y = position[1, triangle]
            coord = np.transpose(np.vstack((x, y)))
            coord = self.translate(coord, position.min(), position.max())

            triangle_color = color[:, triangle]
            triangle_color = (np.average(triangle_color, axis=1)) * 255

            cv2.fillPoly(image, np.int32([coord]), color=tuple([int(x) for x in triangle_color]))

        center = (w / 2, w / 2)
        angle180 = 180
        scale = 1.0

        rotation_matrix = cv2.getRotationMatrix2D(center, angle180, scale)
        rotated180 = cv2.warpAffine(image, rotation_matrix, (w, w))

        return rotated180

    def create_image_and_save(self, n):
        formation, cells = self.get_vectors(n)
        position = formation['position']
        color = formation['color']

        # create image
        image = self.patch(position, color, cells)

        # get face mask without mouth interior
        cut = ld.LandmarkDetection()
        # RGB image with face
        cutout_face = cut.cutout_mask_array(np.uint8(image), True)

        # crop, resize and save face
        cropper = fc.FaceCropper()
        cropper.generate(np.uint8(cutout_face), True, n)
        print(1)


def main():
    preprocess = ImagePreprocess()
    preprocess.create_image_and_save(1)


main()
