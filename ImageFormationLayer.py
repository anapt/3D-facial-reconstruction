import numpy as np
import matlab.engine
import matplotlib.pyplot as plt

import semanticCodeVector as scv
import parametricMoDecoder as pmd
import LandmarkDetection as ld
import FaceCropper as fc


class ImageFormationLayer(object):

    def __init__(self, path, vector):
        self.vector = vector
        self.path = path

    def get_vertices_and_reflectance(self):
        semantic = scv.SemanticCodeVector(self.path)
        vertices = semantic.calculate_coords(self.vector)
        # read average face cells
        cells = semantic.read_cells()
        reflectance = semantic.calculate_reflectance(self.vector)

        return vertices, reflectance, cells

    def get_reconstructed_image(self):
        vertices, reflectance, cells = self.get_vertices_and_reflectance()
        decoder = pmd.ParametricMoDecoder(vertices, reflectance, self.vector, cells)

        image = decoder.get_image_formation()

        cells = decoder.calculate_cell_depth()
        cells = cells.tolist()
        eng = matlab.engine.start_matlab()
        # coords = image['position']
        position = image['position'].tolist()

        color = image['color'].tolist()

        # draw image
        image = eng.patch_and_show(position, color, cells)

        # get face mask without mouth interior
        cut = ld.LandmarkDetection()
        cutout_face = cut.cutout_mask_array(np.uint8(image), False)

        # crop and resize face
        cropper = fc.FaceCropper()
        cropped_face = cropper.generate(np.uint8(cutout_face), False, None)

        return cropped_face


def main():
    show_result = True
    n = 5
    path = './DATASET/model2017-1_bfm_nomouth.h5'
    vector_path = ("./DATASET/semantic/x_%d.txt" % n)
    vector = np.loadtxt(vector_path)

    x = {
        "shape": vector[0:80, ],
        "expression": vector[80:144, ],
        "reflectance": vector[144:224, ],
        "rotation": vector[224:227, ],
        "translation": vector[227:230, ],
        "illumination": vector[230:257, ]
    }

    formation = ImageFormationLayer(path, x)
    image = formation.get_reconstructed_image()

    if show_result:
        plt.imshow(image)
        plt.show()


main()
