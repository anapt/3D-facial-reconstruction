import numpy as np
import matlab.engine

import semanticCodeVector as scv
import ImagePreProcessing as preprocess
import parametricMoDecoder as pmd


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

    def get_reconstructed_image(self, show_result):
        vertices, reflectance, cells = self.get_vertices_and_reflectance()
        decoder = pmd.ParametricMoDecoder(vertices, reflectance, self.vector, cells)

        image = decoder.get_image_formation()

        if show_result:
            cells = decoder.calculate_cell_depth()
            eng = matlab.engine.start_matlab()
            position = image['position'].tolist()

            color = image['color'].tolist()
            # print(color.shape)
            eng.patch_and_show(position, color, cells.tolist(), nargout=0)

        return image


def main():
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
    image = formation.get_reconstructed_image(True)
    # print(image['position'].shape)
    # TODO add preprocess


main()
