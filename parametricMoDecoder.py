import h5py
import numpy as np
import matplotlib.pyplot as plt
import semanticCodeVector as scv

class ParametricMoDecoder:

    def __init__(self, vertices, reflectance):
        self.vertices = vertices
        self.illumination = reflectance

    def projection(self, coords_3d):

        if coords_3d[2] > 60:
            inv_z = 1/coords_3d[2]
            m = np.matrix([[inv_z, 0, 0], [0, inv_z, 0]])
            # coords_2d = np.dot(m, coords_3d)
            coords_2d = np.matrix([coords_3d[0]*inv_z, coords_3d[1]*inv_z])
            return np.transpose(coords_2d)
        else:
            coords_2d = np.transpose(np.matrix([0, 0]))
            return coords_2d






def main():
    # MODIFY TO path containing Basel Face model
    path = '/home/anapt/Documents/Thesis - data/data-raw/model2017-1_bfm_nomouth.h5'
    scv1 = scv.SemanticCodeVector(path)
    x = scv1.sample_vector()

    # print(x["illumination"])

    # scv.plot_face3d()
    vertices = scv1.calculate_coords(x)
    reflectance = scv1.calculate_reflectance(x)

    scv_pca_bases = scv1.read_pca_bases()

    vertices = np.asarray(scv_pca_bases["average_shape"])
    ver = np.reshape(vertices, (3, int(vertices.size / 3)), order='F')

    pmod = ParametricMoDecoder(vertices, reflectance)
    # print(pmod.vertices.shape)
    # ver = np.reshape(pmod.vertices, (3, int(pmod.vertices.size / 3)), order='F')
    # print(ver.shape)
    # v0 = ver[:, 0]
    projected = np.zeros([2, 53149])
    print(projected.shape)
    for i in range(0, 53149):
        projected[:, i] = pmod.projection(ver[:, i]).ravel()
        # print(projected[:, i].shape)
        # print(pmod.projection(ver[:, i]).shape)
        # print(pmod.projection(ver[:, i]).ravel())

    print(projected.max())
    plt.scatter(projected[0, ], projected[1, ], s=1)
    plt.show()
    # plt.plot(vertices[2,: ])
    # plt.show()
    # pmod.projection(v0)


main()