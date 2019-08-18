import numpy as np
import math as math
from FaceNet3D import FaceNet3D as Helpers


class ParametricMoDecoder(Helpers):

    def __init__(self, vertices, color, x, cells):
        """
        Class initializer

        :param vertices: <class 'numpy.ndarray'>    (159447,)
        :param color: <class 'numpy.ndarray'> (159447,)
        :param x: Semantic Code Vector              dictionary
        :param cells: <class 'numpy.ndarray'>       (3, 105694)
        """
        super().__init__()
        self.vertices = vertices
        self.color = color
        self.x = x
        self.cells = cells

    @staticmethod
    def projection(coords_3d):
        """
        Projects coordinates from camera space to screen space

        :param coords_3d: 3D coordinates in CCS shape : (3, 53149)
        :return: 2D coordinates in Screen space with shape (2, 53149)
        """
        coords_2d = np.zeros((2, coords_3d.shape[1]), dtype=coords_3d.dtype)
        for i in range(0, coords_3d.shape[1]):
            if (coords_3d[2, i]) > 0:
                inv_z = (1/coords_3d[2, i])
                coords_2d[:, i] = ([coords_3d[0, i]*inv_z, coords_3d[1, i]*inv_z])

            else:
                coords_2d[:, i] = [0, 0]

        return coords_2d

    @staticmethod
    def create_rot_mat(a, b, c):
        """
        Creates rotation matrix from yaw-pitch-roll angles

        :param a: yaw angle - rotation around the x-axis (degrees)
        :param b: pitch angle - rotation around the y-axis (degrees)
        :param c: roll angle - rotation around the z-axis (degrees)
        :return: <class 'numpy.matrix'> with shape (3,3) (SO3 rotation matrix)
        """
        a = 10 * a
        b = 10 * b
        c = 10 * c

        c1 = math.cos(math.radians(a))
        c2 = math.cos(math.radians(b))
        c3 = math.cos(math.radians(c))

        s1 = math.sin(math.radians(a))
        s2 = math.sin(math.radians(b))
        s3 = math.sin(math.radians(c))

        rot_mat_so3 = np.matrix([[c1*c2, c1*s2*s3 - s1*c3, c1*s2*c3 + s1*s3],
                                 [s1*c2, s1*s2*s3 + c1*c3, s1*s2*c3 - c1*s3],
                                 [-s2, c2*s3, c2*c3]])

        return rot_mat_so3

    def transform_wcs2ccs(self, coords_ws, inv_rotmat, translation):
        """
        Affine transformation from World Space Coordinate to Camera Space Coordinates

        :param coords_ws: coordinates in WCS with shape (3, 53149)
        :param inv_rotmat: inverse of rotation matrix
        :param translation: translation vector (part of the Semantic Code Vector)
        :return: coordinates in CCS with shape (3, 53149)
        """
        coords = np.matmul(inv_rotmat, coords_ws)
        coords_cs = coords - np.reshape(np.transpose(translation), newshape=(3, 1))

        return coords_cs

    def get_image_formation(self):
        """
        Function that:
                        reshapes vertices and reflectance to shape (3, N) where N the number of vertices
                        gets Rotation matrix and inverses it
                        gets World Space Normals
                        transforms World Space Normals to Camera Space Normals
                        gets color for each vertex
                        transforms vertices coordinates from WCS to CCS
                        projects CCS coordinates to screen space

        :return: dictionary with keys   position    <class 'numpy.ndarray'> (2, 53149) (projected vertices coordinates)
                                        color       <class 'numpy.ndarray'> (3, 53149) (color of vertices)
        """
        ws_vertices = np.reshape(self.vertices, (3, int(self.vertices.size / 3)), order='F')
        color = np.reshape(self.color, (3, int(self.color.size / 3)), order='F')
        # print(np.ceil(reflectance*255))

        rotmat_so3 = self.create_rot_mat(self.x['rotation'][0], self.x['rotation'][1], self.x['rotation'][2])

        # Calculate projected coordinates
        translation = np.array([0, 0, -5000])
        cs_vertices = self.transform_wcs2ccs(ws_vertices, np.linalg.inv(rotmat_so3), translation)
        projected = self.projection(cs_vertices)

        formation = {
            "position": projected,
            "color": color
        }

        return formation

    def calculate_cell_depth(self):
        """
        Calculates depth of each triangle and returns the 50000 triangles with smallest depth

        :return: <class 'numpy.ndarray'> with shape (3, 50000)
        """
        vertices = np.reshape(self.vertices, (3, int(self.vertices.size / 3)), order='F')
        depth = np.zeros(self.cells.shape[1], dtype=self.cells.dtype)
        for i in range(0, self.cells.shape[1]):
            depth[i] = (vertices[2, self.cells[0, i]] + vertices[2, self.cells[1, i]] +
                        vertices[2, self.cells[2, i]]) / 3

        # arrange cells with deepest one first
        order = np.argsort(depth)
        cells_ordered = self.cells[:, order.astype(int)]

        # cells_ordered = cells_ordered[:, (cells_ordered.shape[1]-50000):]

        return cells_ordered
