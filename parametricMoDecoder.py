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
    def projection(coords_3d):
        # print(coords_3d.shape)
        coords_2d = np.zeros((2, coords_3d.shape[1]), dtype=coords_3d.dtype)

        for i in range(0, coords_3d.shape[1]):
            if abs(coords_3d[2, i]) > 70:
                inv_z = abs(1/coords_3d[2, i])
                # m = np.matrix([[inv_z, 0, 0], [0, inv_z, 0]])
                # coords_2d = np.dot(m, coords_3d)
                coords_2d[:, i] = ([coords_3d[0, i]*inv_z, coords_3d[1, i]*inv_z])

            else:
                coords_2d[:, i] = [0, 0]

        return np.transpose(coords_2d)

    @staticmethod
    def create_rot_mat(a, b, c):
        # a = b = c = 0
        # no rotation for testing
        a = 0
        b = 0
        c = 0
        c1 = math.cos(math.radians(c))
        c2 = math.cos(math.radians(b))
        c3 = math.cos(math.radians(a))

        s1 = math.sin(math.radians(c))
        s2 = math.sin(math.radians(b))
        s3 = math.sin(math.radians(a))

        rot_mat_so3 = np.matrix([[c1*c2, c1*s2*s3 - c3*s1, s1*s3 - c1*c3*s2],
                                 [c2*s1, c1*c3 + s1*s2*s3, c3*s1*s2 - c1*c3],
                                 [-s2, c2*s3, c2*c3]])
        # print(type(rot_mat_so3))
        # print(rot_mat_so3.shape)
        return rot_mat_so3

    @staticmethod
    def transform_wcs2ccs(coords_ws, inv_rotmat, translation):
        print(np.shape(coords_ws))

        coords_cs = np.zeros(coords_ws.shape, dtype=coords_ws.dtype)
        for i in range(0, coords_ws.shape[1]):
            coords_cs[::, i] = np.dot(inv_rotmat, (coords_ws[::, i] - translation))

        return coords_cs

    @staticmethod
    def get_sh_basis_function(x_coord, y_coord, z_coord, b):
        r = pow(x_coord, 2) + pow(y_coord, 2) + pow(z_coord, 2)
        if b == 9:
            basis = 0.75 * math.pow(35 / math.pi, 0.5) * (x_coord * y_coord *
                                                          (pow(x_coord, 2) - pow(y_coord, 2))) / (pow(r, 4))
        elif b == 7:
            basis = 0.75 * math.pow(35 / (2 * math.pi), 0.5) * \
                 (z_coord * y_coord * (3 * pow(x_coord, 2) - pow(y_coord, 2))) / (pow(r, 4))
        elif b == 5:
            basis = 0.75 * math.pow(5 / math.pi, 0.5) * (x_coord * y_coord *
                                                         (7 * pow(z_coord, 2) - pow(r, 2))) / (pow(r, 4))
        elif b == 3:
            basis = 0.75 * math.pow(5 / (2 * math.pi), 0.5) * \
                 (y_coord * z_coord * (7 * pow(z_coord, 2) - 3 * pow(r, 2))) / (pow(r, 4))
        elif b == 1:
            basis = (3 / 16) * math.pow(1 / math.pi, 0.5) * \
                 (35 * pow(z_coord, 2) - 30 * pow(z_coord, 2) * pow(r, 2) + 3 * pow(r, 4)) / (pow(r, 4))
        elif b == 2:
            basis = 0.75 * math.pow(5 / (2 * math.pi), 0.5) * \
                 (x_coord * z_coord * (7 * pow(z_coord, 2) - 3 * pow(r, 2))) / (pow(r, 4))
        elif b == 4:
            basis = (3 / 8) * math.pow(5 / math.pi, 0.5) * ((pow(x_coord, 2) - pow(y_coord, 2)) *
                                                            (7 * pow(z_coord, 2) - pow(r, 2))) / (pow(r, 4))
        elif b == 6:
            basis = (3 / 4) * math.pow(35 / (2 * math.pi), 0.5) * \
                    ((pow(x_coord, 2) - 3 * pow(y_coord, 2)) * x_coord * y_coord) / (pow(r, 4))
        elif b == 8:
            basis = (3 / 16) * math.pow(35 / math.pi, 0.5) * \
                    ((pow(x_coord, 2)) * (pow(x_coord, 2) - 3 * pow(y_coord, 2)) - (pow(y_coord, 2)) *
                     (3 * pow(x_coord, 2) - pow(y_coord, 2))) / (pow(r, 4))
        return basis

#    illumination model
    def get_color(self, reflectance, normal, illumination):
        illumination = np.reshape(illumination, (3, 9), order='F')
        summ = 0
        for i in range(0, 9):
            summ = summ + illumination[:, i] * self.get_sh_basis_function(normal[0], normal[1], normal[2], i+1)
        # point wise multiplication
        color = np.multiply(reflectance, summ)
        color = np.transpose(color)
        return color

    @staticmethod
    def normalize_v3(arr):
        # ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
        lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
        arr[:, 0] /= lens
        arr[:, 1] /= lens
        arr[:, 2] /= lens
        return arr

    def calculate_normals(self, cells):
        # prepare data
        vertices = self.vertices
        vertices = np.reshape(vertices, (3, int(vertices.size / 3)), order='F')
        vertices = np.transpose(vertices)

        cells = np.transpose(cells)

        # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
        norm = np.zeros(vertices.shape, dtype=vertices.dtype)
        # Create an indexed view into the vertex array using the array of three indices for triangles
        tris = vertices[cells]
        # print(tris.shape) # (105694, 3, 3)

        # Calculate the normal for all the triangles, by taking the cross product of the vectors
        # v1-v0, and v2-v0 in each triangle
        n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
        # n is now an array of normals per triangle
        # we need to normalize these, so that our next step weights each normal equally.
        n = self.normalize_v3(n)
        # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
        # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
        # the triangles' normal. Multiple triangles would then contribute to every vertex,
        # so we need to normalize again afterwards.
        norm[cells[:, 0]] += n
        norm[cells[:, 1]] += n
        norm[cells[:, 2]] += n
        norm = self.normalize_v3(norm)

        norm = np.transpose(norm)  # return 3, 53149 ndarray
        return norm


def main():
    # MODIFY TO path containing Basel Face model
    path = '/home/anapt/Documents/Thesis - data/data-raw/model2017-1_bfm_nomouth.h5'
    data = scv.SemanticCodeVector(path)
    x = data.sample_vector()
    cells = data.read_cells()

    # vertices = scv1.calculate_coords(x)
    reflectance = data.calculate_reflectance(x)

    scv_pca_bases = data.read_pca_bases()

    vertices = np.asarray(scv_pca_bases["average_shape"])
    ver = np.reshape(vertices, (3, int(vertices.size / 3)), order='F')
    ref = np.reshape(reflectance, (3, int(reflectance.size / 3)), order='F')

    pmod = ParametricMoDecoder(vertices, reflectance)

    rotmatSO3 = pmod.create_rot_mat(x['rotation'][0], x['rotation'][1], x['rotation'][2])
    inv_rotmat = np.linalg.inv(rotmatSO3)
    pmod.transform_wcs2ccs(ver[:, 2], inv_rotmat, x['translation'])
    projected = np.zeros([2, 53149])

    pmod.get_color(ref[:, 0], ver[:, 0], x['illumination'])
    # for i in range(0, 53149):
    #     coords_ccs = pmod.transform_wcs2ccs(ver[:, i], inv_rotmat, x["translation"])
    #     projected[:, i] = pmod.projection(coords_ccs)
    #
    # print(projected.max())
    # plt.scatter(projected[0, ], projected[1, ], s=1)
    # plt.show()
    # plt.plot(vertices[2,: ])
    # plt.show()
    # pmod.projection(v0)


# main()