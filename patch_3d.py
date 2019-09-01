import pylab as pl
from FaceNet3D import FaceNet3D as Helpers
import numpy as np
from ParametricMoDecoder import ParametricMoDecoder
from SemanticCodeVector import SemanticCodeVector
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt


class Reconstruction(Helpers):

    def __init__(self):
        """
        Class initializer
        """
        super().__init__()
        self.data = SemanticCodeVector()

    def get_vectors(self, vector):
        """
        Samples vector, saves vector in .txt file
        Calculate image formation (2d coordinates and color)

        :param vector: <class 'numpy.ndarray'> with shape (self.scv_length)
        :return:    vertices    3D coordinates of the vertices /in [-1, 1]
                                <class 'numpy.ndarray'> with shape (3, self.num_of_vertices)
                    color       color of the vertices /in [-1, 1]
                                <class 'numpy.ndarray'> with shape (3, self.num_of_vertices)
                    cells       connections between vertices
                                <class 'numpy.ndarray'> with shape (3, self.num_of_cells)
        """

        cells = self.data.read_cells()

        x = self.vector2dict(vector)

        vertices = self.data.calculate_3d_vertices(x)
        color = self.data.calculate_color(x)

        decoder = ParametricMoDecoder(vertices, color, x, cells)

        cells_ordered = decoder.calculate_cell_depth()

        vertices = np.reshape(vertices, (3, int(vertices.size / 3)), order='F')
        vertices = vertices / np.amax(vertices)
        color = np.reshape(color, (3, int(color.size / 3)), order='F')

        return vertices, color, cells_ordered

    @staticmethod
    def patch_3d(vertices, color, cells):
        """
        Drawing function

        :param vertices:    3D coordinates of the vertices
                            <class 'numpy.ndarray'> with shape (3, self.num_of_vertices)
        :param color:       color of the vertices
                            <class 'numpy.ndarray'> with shape (3, self.num_of_vertices)
        :param cells:       connections between vertices
                            <class 'numpy.ndarray'> with shape (3, self.num_of_cells)
        :return:
        """
        n_cells = cells.shape[1]

        fig = plt.figure()
        ax = Axes3D(fig, azim=75, elev=0)
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)

        for i in range(0, n_cells):
            triangle = cells[:, i]
            x = vertices[0, triangle]
            y = vertices[2, triangle]
            z = vertices[1, triangle]
            verts = [list(zip(x, y, z))]
            tri_color = color[:, triangle]
            triangle_color = (np.average(tri_color, axis=1))
            ax.add_collection3d(Poly3DCollection(verts, facecolors=triangle_color, alpha=1.0), zs='z')

        pl.show()


def main():
    plot = Reconstruction()
    x = np.zeros((231,))
    ver, col, cel = plot.get_vectors(x)
    plot.patch_3d(ver, col, cel)


main()
