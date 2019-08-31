import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl
import scipy as sp
from FaceNet3D import FaceNet3D as Helpers
import numpy as np
from ParametricMoDecoder import ParametricMoDecoder
from SemanticCodeVector import SemanticCodeVector
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matlab.engine
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter



class Reconstruction(Helpers):

    def __init__(self):
        super().__init__()
        self.data = SemanticCodeVector()

    def get_vectors(self, vector):
        """
        Samples vector, saves vector in .txt file
        Calculate image formation (2d coordinates and color)

        :param n: iteration number
        :return:    image formation (dictionary with keys position, color)
                    cell ordered with deepest one first
        """

        cells = self.data.read_cells()

        # x = data.sample_vector()
        x = self.vector2dict(vector)

        vertices = self.data.calculate_3d_vertices(x)
        color = self.data.calculate_color(x)

        decoder = ParametricMoDecoder(vertices, color, x, cells)

        cells_ordered = decoder.calculate_cell_depth()

        vertices = np.reshape(vertices, (3, int(vertices.size / 3)), order='F')
        vertices = vertices / np.amax(vertices)
        color = np.reshape(color, (3, int(color.size / 3)), order='F')

        print("vertices shape", vertices.shape)
        print("color shape", color.shape)
        return vertices, color, cells_ordered

    def rotate(self, ax):
        for angle in range(0, 45):
            ax.view_init(10, angle)
            return plt.draw()
            # plt.pause(0.1)

    def patch_3d(self, vertices, color, cells):
        """
        Drawing function

        :param position:    projected coordinates of the vertices
                            <class 'numpy.ndarray'> with shape (2, 53149)
        :param color:       color of the vertices
                            <class 'numpy.ndarray'> with shape (3, 53149)
        :param cells:       array containing the connections between vertices
                            <class 'numpy.ndarray'> with shape (3, 50000)
        :return:            drawn image
                            <class 'numpy.ndarray'> with shape (500, 500, 3)
        """
        n_cells = cells.shape[1]
        # np.savetxt("./cells.txt", cells)
        # w = 500
        # image = np.zeros((w, w, 3), dtype=np.uint8)
        #
        # coord = np.zeros(shape=(3, 2, n_cells))
        #
        # for i in range(0, n_cells):
        #     triangle = cells[:, i]
        #     x = position[0, triangle]
        #     y = position[1, triangle]
        #     coord[:, :, i] = np.transpose(np.vstack((x, y)))
        #
        # coord = self.translate(coord, np.amin(coord), np.amax(coord), 130, 370)
        #
        # for i in range(0, n_cells):
        #     triangle = cells[:, i]
        #
        #     tri_color = color[:, triangle]
        #     triangle_color = (np.average(tri_color, axis=1)) * 255
        #
        #     cv2.fillConvexPoly(image, np.int64([coord[:, :, i]]), color=tuple([int(x) for x in triangle_color]))
        # plt.clf()
        fig = plt.figure()
        ax = Axes3D(fig, azim=75, elev=0)
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)
        # verts = [list(zip(x, y, z))]
        # print(verts)
        # ax.add_collection3d(Poly3DCollection(verts), zs='z')
        # plt.show()
        for i in range(0, n_cells):
            triange = cells[:, i]
            x = vertices[0, triange]
            y = vertices[2, triange]
            z = vertices[1, triange]
            # x = [0, 1, 1, 0]
            # y = [0, 0, 1, 1]
            # z = [0, 1, 0, 1]
            verts = [list(zip(x, y, z))]
            # print(verts)
            # vtx = vertices[:, triange]
            # vtx = sp.rand(3, 3)
            tri_color = color[:, triange]
            triangle_color = (np.average(tri_color, axis=1))
            ax.add_collection3d(Poly3DCollection(verts, facecolors=triangle_color, alpha=1.0), zs='z')

        # for angle in range(0, 360):
        #     ax.view_init(90, angle)
        #     plt.draw()
        #     plt.pause(0.01)

        # ani = animation.FuncAnimation(fig, self.rotate(ax), 25,
        #                                    interval=50, blit=False)
        # ani = animation.FuncAnimation(
        #     fig, animate, init_func=init, interval=2, blit=True, save_count=50)
        # ani.save("movie.mp4")
        #
        # or
        #

        # writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        # ani.save("movie.mp4", writer=writer)

        pl.show()



def main():
    plot = Reconstruction()
    x = np.zeros((231,))
    ver, col, cel = plot.get_vectors(x)
    plot.patch_3d(ver, col, cel)

main()