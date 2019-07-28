import numpy as np
import matplotlib.pyplot as plt

import SemanticCodeVector as scv
import ParametricMoDecoder as pmd
import LandmarkDetection as ld
from unused import FaceCropper as fc
import ImagePreprocess as preprocess


class ImageFormationLayer(object):

    def __init__(self, vector):
        self.vector = self.vector2dict(vector)
        self.path = './DATASET/model2017-1_bfm_nomouth.h5'
        self.preprocess = preprocess.ImagePreprocess()

    def get_vertices_and_reflectance(self):
        semantic = scv.SemanticCodeVector(self.path)
        vertices = semantic.calculate_coords(self.vector)
        reflectance = semantic.calculate_reflectance(self.vector)

        # read average face cells
        cells = semantic.read_cells()

        return vertices, reflectance, cells

    def get_reconstructed_image(self):
        vertices, reflectance, cells = self.get_vertices_and_reflectance()
        decoder = pmd.ParametricMoDecoder(vertices, reflectance, self.vector, cells)

        formation = decoder.get_image_formation()
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

        # crop and resize face
        # cropper = fc.FaceCropper()
        # cropped_face = cropper.generate(np.uint8(cutout_face), False, None)

        return cutout_face

    def get_reconstructed_image_for_loss(self):
        vertices, reflectance, cells = self.get_vertices_and_reflectance()
        decoder = pmd.ParametricMoDecoder(vertices, reflectance, self.vector, cells)

        formation = decoder.get_image_formation()
        cells = decoder.calculate_cell_depth()

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
        cutout_face = cut.cutout_mask_array(np.uint8(image), False)

        # crop and resize face
        cropper = fc.FaceCropper()
        cropped_face = cropper.generate(np.uint8(cutout_face), False, None)

        return cropped_face, indices, position

    def get_sampled_indices(self):
        vertices, reflectance, cells = self.get_vertices_and_reflectance()
        decoder = pmd.ParametricMoDecoder(vertices, reflectance, self.vector, cells)

        cells = decoder.calculate_cell_depth()

        indices = np.unique(cells)
        indices = np.random.choice(indices, size=13000, replace=False)

        return indices

    @staticmethod
    def vector2dict(vector):
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
    show_result = True
    n = 0
    vector_path = ("./DATASET/semantic/x_{:06}.txt".format(n))
    vector = np.loadtxt(vector_path)

    formation = ImageFormationLayer(vector)
    image = formation.get_reconstructed_image()

    if show_result:
        plt.imshow(image)
        plt.show()


# main()
