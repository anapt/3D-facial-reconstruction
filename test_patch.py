import cv2

import numpy as np
import matlab.engine

import parametricMoDecoder as pmd
import semanticCodeVector as scv
import LandmarkDetection as ld
import FaceCropper as fc


def get_vectors(path, n):
    """
    Samples vector, saves vector in .txt file
    Calculate image formation (2d coordinates and color)

    :param path: Path to Basel Face Model
    :param n: iteration number
    :return:    image formation (dictionary with keys position, color)
                cell ordered with deepest one first
    """
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
    # np.savetxt("./DATASET/color/color_%d.txt" % n, formation['color'])
    # np.savetxt("./DATASET/position/position_%d.tgxt" % n, formation['position'])

    cells_ordered = decoder.calculate_cell_depth()
    # np.savetxt("DATASET/cells/cells_%d.txt" % n, cells_ordered)

    return formation, cells_ordered


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


def patch(position, color, cells):
    n_cells = 105694
    w = 500
    image = np.zeros((w, w, 3), dtype=np.uint8)

    for i in range(0, n_cells):
        triangle = cells[:, i]
        x = position[0, triangle]
        y = position[1, triangle]
        coord = np.transpose(np.vstack((x, y)))
        coord = translate(coord, position.min(), position.max())

        triangle_color = color[:, triangle]
        triangle_color = (np.average(triangle_color, axis=1)) * 255

        cv2.fillPoly(image, np.int32([coord]), color=tuple([int(x) for x in triangle_color]))

    center = (w/2, w/2)
    angle180 = 180
    scale = 1.0

    rotation_matrix = cv2.getRotationMatrix2D(center, angle180, scale)
    rotated180 = cv2.warpAffine(image, rotation_matrix, (w, w))

    # cv2.imshow('Image rotated by 180 degrees', rotated180)
    # cv2.waitKey(0)  # waits until a key is pressed
    # cv2.destroyAllWindows()  # destroys the window showing image

    return rotated180


def main():
    # Number of images to create
    N = 1000
    path = './DATASET/model2017-1_bfm_nomouth.h5'

    for n in range(0, 1):
        formation, cells = get_vectors(path, n)
        position = formation['position']
        color = formation['color']
        cells = cells

        # create image
        image = patch(position, color, cells)

        # get face mask without mouth interior
        cut = ld.LandmarkDetection()
        cutout_face = cut.cutout_mask_array(np.uint8(image), True)

        # crop and resize face
        cropper = fc.FaceCropper()
        cropper.generate(np.uint8(cutout_face), True, n)
        print(n)


main()
