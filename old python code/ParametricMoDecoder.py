import numpy as np
import math as math
import ImagePreprocess as preprocess
import time

class ParametricMoDecoder:

    def __init__(self, vertices, reflectance, x, cells):
        """
        Class initializer

        :param vertices: <class 'numpy.ndarray'>    (159447,)
        :param reflectance: <class 'numpy.ndarray'> (159447,)
        :param x: Semantic Code Vector              dictionary
        :param cells: <class 'numpy.ndarray'>       (3, 105694)
        """
        self.vertices = vertices
        self.reflectance = reflectance
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
            if (coords_3d[2, i]) >= 1:
                inv_z = (1/coords_3d[2, i])
                coords_2d[:, i] = ([coords_3d[0, i]*inv_z, coords_3d[1, i]*inv_z])

            else:
                coords_2d[:, i] = [0, 0]

        translate = preprocess.ImagePreprocess()
        coords_2d = translate.translate(coords_2d, np.amin(coords_2d), np.amax(coords_2d), 1, 500)
        return coords_2d

    @staticmethod
    def create_rot_mat(a, b, c):
        """
        Creates rotation matrix from yaw-pitch-roll angles

        :param a: yaw angle (degrees)
        :param b: pitch angle (degrees)
        :param c: roll angle (degrees)
        :return: <class 'numpy.matrix'> with shape (3,3) (SO3 rotation matrix)
        """
        c1 = math.cos(math.radians(c))
        c2 = math.cos(math.radians(b))
        c3 = math.cos(math.radians(a))

        s1 = math.sin(math.radians(c))
        s2 = math.sin(math.radians(b))
        s3 = math.sin(math.radians(a))

        rot_mat_so3 = np.matrix([[c1*c2, c1*s2*s3 - c3*s1, s1*s3 - c1*c3*s2],
                                 [c2*s1, c1*c3 + s1*s2*s3, c3*s1*s2 - c1*s3],
                                 [-s2, c2*s3, c2*c3]])

        return rot_mat_so3

    @staticmethod
    def transform_wcs2ccs(coords_ws, inv_rotmat, translation):
        """
        Affine transformation from World Space Coordinate to Camera Space Coordinates

        :param coords_ws: coordinates in WCS with shape (3, 53149)
        :param inv_rotmat: inverse of rotation matrix
        :param translation: translation vector (part of the Semantic Code Vector)
        :return: coordinates in CCS with shape (3, 53149)
        """
        translate = preprocess.ImagePreprocess()
        coords_ws = translate.translate(coords_ws, np.amin(coords_ws), np.amax(coords_ws), 1, 500)

        coords_cs = np.matmul(inv_rotmat, np.transpose(np.transpose(coords_ws) - (100 * translation)))

        return coords_cs

    @staticmethod
    def transform_wcs2ccs_vectors(coords_ws, inv_rotmat, translation):
        """
        Affine transformation from World Space Coordinate to Camera Space Coordinates

        :param coords_ws: coordinates in WCS with shape (3, 53149)
        :param inv_rotmat: inverse of rotation matrix
        :param translation: translation vector (part of the Semantic Code Vector)
        :return: coordinates in CCS with shape (3, 53149)
        """
        coords_cs = np.zeros(coords_ws.shape, dtype=coords_ws.dtype)

        for i in range(0, coords_ws.shape[1]):
            coords_cs[::, i] = np.dot(inv_rotmat, (coords_ws[::, i] - (10 * translation)))

        return coords_cs

    # # TODO check if that's correct
    # @staticmethod
    # def transform_wc2cs_vectors(coords_ws, rotmat):
    #     coords_cs = np.zeros(coords_ws.shape, dtype=coords_ws.dtype)
    #     for i in range(0, coords_ws.shape[1]):
    #         coords_cs[::, i] = np.dot(rotmat, coords_ws[::, i])
    #
    #     return coords_cs

    @staticmethod
    def get_sh_basis_function(x_coord, y_coord, z_coord, b):
        """
        Fourth order spherical harmonics

        :param x_coord: x coordinate
        :param y_coord: y coordinate
        :param z_coord: z coordinate
        :param b: parameters that specifies iteration number
        :return: scalar (float)
        """
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

    @staticmethod
    def spherical_harmonics(x, y, z, b):
        """
        Second order Spherical Harmonics

        :param x: x coordinate
        :param y: x coordinate
        :param z: x coordinate
        :param b: parameters that specifies iteration number
        :return: scalar (float)
        """
        r = pow(x, 2) + pow(y, 2) + pow(z, 2)
        if b == 0:
            n = 0.25 * pow(5 / math.pi, 0.5) * (2 * pow(z, 2) - pow(x, 2) - pow(y, 2)) / r
        elif b == 1:
            n = 0.5 * pow(15 / math.pi, 0.5) * (z*x) / pow(r, 2)
        elif b == 3:
            n = 0.5 * pow(15 / math.pi, 0.5) * (pow(x, 2) - pow(y, 2)) / pow(r, 2)
        elif b == 2:
            n = 0.5 * pow(15 / math.pi, 0.5) * (y*z) / pow(r, 2)
        elif b == 4:
            n = 0.5 * pow(15 / math.pi, 0.5) * (x*y) / pow(r, 2)
        else:
            n = 1
        return n

    @staticmethod
    def spherical_harmonics_array(x, y, z, b):
        """
        Second order Spherical Harmonics

        :param x: x coordinate [1, 3N]
        :param y: x coordinate
        :param z: x coordinate
        :param b: parameters that specifies iteration number
        :return: scalar (float)
        """
        r = np.power(x, 2) + np.power(y, 2) + np.power(z, 2)
        if b == 0:
            n = 0.25 * pow(5 / math.pi, 0.5) * np.divide((2 * np.power(z, 2) - np.power(x, 2) - np.power(y, 2)), r)
        elif b == 1:
            n = 0.5 * pow(15 / math.pi, 0.5) * np.divide((np.multiply(z, x)), np.power(r, 2))
        elif b == 3:
            n = 0.5 * pow(15 / math.pi, 0.5) * np.divide((np.power(x, 2) - np.power(y, 2)), np.power(r, 2))
        elif b == 2:
            n = 0.5 * pow(15 / math.pi, 0.5) * np.divide((np.multiply(y, z)), np.power(r, 2))
        elif b == 4:
            n = 0.5 * pow(15 / math.pi, 0.5) * np.divide((np.multiply(x, y)), np.power(r, 2))
        else:
            n = 1
        return n

    def get_color(self, reflectance, normal, illumination):
        """
        Calculates color of vertex

        :param reflectance: vector of reflectance at vertex with shape (3,)
        :param normal: normal vector at vertex with shape (3,)
        :param illumination: illumination vector from Semantic Code Vector with shape (3, 9)
        :return: <class 'numpy.ndarray'> with shape (3,)
        """
        # sum over illumination and Spherical harmonic scalar
        _sum = np.zeros(shape=(3, 53149))

        for i in range(0, 9):
            _sum = _sum + np.multiply(np.reshape(illumination[:, i], (3, 1)),
                                      self.spherical_harmonics_array(normal[0, :],
                                                                     normal[1, :], normal[2, :], i))
        # point wise multiplication with reflectance at vertex
        color = np.multiply(reflectance, _sum)

        return color

    @staticmethod
    def normalize_v3(arr):
        """
        Normalize a numpy array of 3 component vectors with shape (n,3)

        :param arr: numpy array to be normalized shape (n,3)
        :return: normalized array, same shape as input array
        """
        lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
        arr[:, 0] /= lens
        arr[:, 1] /= lens
        arr[:, 2] /= lens
        return arr

    def calculate_normals(self, cells):
        """
        Function that calculates normals for each vertex

        :param cells: triangles
        :return: <class 'numpy.ndarray'> with shape (3, 53149)
        """
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
        # print(ws_vertices)
        reflectance = np.reshape(self.reflectance, (3, int(self.reflectance.size / 3)), order='F')
        # print(np.ceil(reflectance*255))
        rotmatSO3 = self.create_rot_mat(self.x['rotation'][0], self.x['rotation'][1], self.x['rotation'][2])
        inv_rotmat = np.transpose(rotmatSO3)

        # Calculate color
        # world space normals
        ws_normals = self.calculate_normals(self.cells)

        # transform world space normals to camera space normals
        cs_normals = self.transform_wcs2ccs(ws_normals, inv_rotmat, self.x['translation'])

        # reshape illumination to nd.array with shape (3,9)
        illumination = np.reshape(self.x['illumination'], (3, 9), order='F')
        # color = self.get_color(reflectance, cs_normals, illumination)
        color = reflectance
        # Calculate projected coordinates
        cs_vertices = self.transform_wcs2ccs(ws_vertices, inv_rotmat, self.x['translation'])
        projected = self.projection(cs_vertices)
        # print(color)
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
