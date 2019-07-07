import numpy as np

import ImagePreProcessing as preprocess
import parametricMoDecoder as pmd
import semanticCodeVector as scv


def get_vectors(path, n):
    data = scv.SemanticCodeVector(path)
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
    np.savetxt("./DATASET/color/color_%d.txt" % n, formation['color'])
    np.savetxt("./DATASET/position/position_%d.txt" % n, formation['position'])

    cells_ordered = decoder.calculate_cell_depth()
    np.savetxt("DATASET/cells/cells_%d.txt" % n, cells_ordered)


def prepare_images(n):
    image_path = ("./DATASET/images/im_%d.png" % n)
    cutout_path = ("./DATASET/images/cutout/im_%d.png" % n)
    cropped_image_path = ("./DATASET/images/cropped/image_%d.png" % n)

    prep = preprocess.ImagePreProcessing(n)
    prep.detect_crop_save()


def main():
    # part 1
    path = './DATASET/model2017-1_bfm_nomouth.h5'

    # for n in range(0, 2500):
    #     get_vectors(path, n)
    #     print(n)

    # part 2
    """ run matlab code to generate images """

    # part 3
    for n in range(0, 5):
        preprocess.ImagePreProcessing(n).detect_crop_save()
        print(n)


main()
