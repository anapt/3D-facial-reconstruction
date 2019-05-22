import h5py
import numpy as np
import matplotlib.pyplot as plt
import semanticCodeVector as scv
import math as math


class ParametricMoDecoder:

    def __init__(self, vertices, reflectance):
        self.vertices = vertices
        self.illumination = reflectance

    @staticmethod
    def projection(self, coords_3d):

        if abs(coords_3d[2]) > 70:
            inv_z = abs(1/coords_3d[2])
            # m = np.matrix([[inv_z, 0, 0], [0, inv_z, 0]])
            # coords_2d = np.dot(m, coords_3d)
            coords_2d = [coords_3d[0]*inv_z, coords_3d[1]*inv_z]

            return np.transpose(coords_2d)
        else:
            coords_2d = [0, 0]
            return np.transpose(coords_2d)

    @staticmethod
    def create_rot_mat(self, a, b, c):
        c1 = math.cos(math.radians(c))
        c2 = math.cos(math.radians(b))
        c3 = math.cos(math.radians(a))

        s1 = math.sin(math.radians(c))
        s2 = math.sin(math.radians(b))
        s3 = math.sin(math.radians(a))

        rot_mat_so3 = np.matrix([[c1*c2, c1*s2*s3 - c3*s1, s1*s3 - c1*c3*s2],
                                 [c2*s1, c1*c3 + s1*s2*s3, c3*s1*s2 - c1*c3],
                                 [-s2, c2*s3, c2*c3]])

        return rot_mat_so3

    @staticmethod
    def transform_wcs2ccs(self, coords_ws, inv_rotmat, translation):
        coords_cs = np.dot(inv_rotmat, (coords_ws - translation))

        return np.transpose(coords_cs)

    @staticmethod
    def get_sh_basis_function(self, x_coord, y_coord, z_coord, b):
        r = x_coord ^ 2 + y_coord ^ 2 + z_coord ^ 2
        if b == 1:
            basis = 0.75 * math.pow(35 / math.pi, 0.5) * (x_coord * y_coord * (x_coord ^ 2 - y_coord ^ 2)) / (r ^ 4)
        elif b == 2:
            basis = 0.75 * math.pow(35 / (2 * math.pi), 0.5) * \
                 (z_coord * y_coord * (3 * x_coord ^ 2 - y_coord ^ 2)) / (r ^ 4)
        elif b == 3:
            basis = 0.75 * math.pow(5 / math.pi, 0.5) * (x_coord * y_coord * (7 * z_coord ^ 2 - r ^ 2)) / (r ^ 4)
        elif b == 4:
            basis = 0.75 * math.pow(5 / (2 * math.pi), 0.5) * \
                 (y_coord * z_coord * (7 * z_coord ^ 2 - 3 * r ^ 2)) / (r ^ 4)
        elif b == 5:
            basis = (3 / 16) * math.pow(1 / math.pi, 0.5) * \
                 (35 * z_coord ^ 2 - 30 * z_coord ^ 2 * r ^ 2 + 3 * r ^ 4) / (r ^ 4)
        elif b == 6:
            basis = 0.75 * math.pow(5 / (2 * math.pi), 0.5) * \
                 (x_coord * z_coord * (7 * z_coord ^ 2 - 3 * r ^ 2)) / (r ^ 4)
        elif b == 7:
            basis = (3 / 8) * math.pow(5 / math.pi, 0.5) * ((x_coord ^ 2 - y_coord ^ 2) *
                                                         (7 * z_coord ^ 2 - r ^ 2)) / (r ^ 4)
        elif b == 8:
            basis = (3 / 4) * math.pow(35 / (2 * math.pi), 0.5) * ((x_coord ^ 2 -
                                                                 3 * y_coord ^ 2) * x_coord * y_coord) / (r ^ 4)
        elif b == 9:
            basis = (3 / 16) * math.pow(35 / math.pi, 0.5) * ((x_coord ^ 2) * (x_coord ^ 2 - 3 * y_coord ^ 2) -
                                                           (y_coord ^ 2) * (3 * x_coord ^ 2 - y_coord ^ 2)) / (r ^ 4)
        return basis


def main():
    # MODIFY TO path containing Basel Face model
    path = '/home/anapt/Documents/Thesis - data/data-raw/model2017-1_bfm_nomouth.h5'
    scv1 = scv.SemanticCodeVector(path)
    x = scv1.sample_vector()

    # vertices = scv1.calculate_coords(x)
    reflectance = scv1.calculate_reflectance(x)

    scv_pca_bases = scv1.read_pca_bases()

    vertices = np.asarray(scv_pca_bases["average_shape"])
    ver = np.reshape(vertices, (3, int(vertices.size / 3)), order='F')

    pmod = ParametricMoDecoder(vertices, reflectance)

    rotmatSO3 = pmod.create_rot_mat(x['rotation'][0], x['rotation'][1], x['rotation'][2])
    inv_rotmat = np.linalg.inv(rotmatSO3)

    projected = np.zeros([2, 53149])

    for i in range(0, 53149):
        coords_ccs = pmod.transform_wcs2ccs(ver[:, i], inv_rotmat, x["translation"])
        projected[:, i] = pmod.projection(coords_ccs)

    print(projected.max())
    plt.scatter(projected[0, ], projected[1, ], s=1)
    plt.show()
    # plt.plot(vertices[2,: ])
    # plt.show()
    # pmod.projection(v0)


main()