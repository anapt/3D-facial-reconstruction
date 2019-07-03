import numpy as np
import math as math


class ParametricMoDecoder:

    def __init__(self, vertices, reflectance, x, cells):
        self.vertices = vertices
        self.reflectance = reflectance
        self.x = x
        self.cells = cells

    ''' Project from camera space to screen space '''
    @staticmethod
    def projection(coords_3d):
        # print(coords_3d.shape)
        coords_2d = np.zeros((2, coords_3d.shape[1]), dtype=coords_3d.dtype)

        for i in range(0, coords_3d.shape[1]):
            if abs(coords_3d[2, i]) > 50:
                inv_z = abs(1/coords_3d[2, i])
                coords_2d[:, i] = ([coords_3d[0, i]*inv_z, coords_3d[1, i]*inv_z])

            else:
                coords_2d[:, i] = [0, 0]
        # shape 2,N
        return coords_2d

    ''' Create rotation matrix from yaw-pitch-roll angles '''
    @staticmethod
    def create_rot_mat(a, b, c):
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

    ''' Affine transformation (points) from World Space Coordinates to Camera Space Coordinates '''
    @staticmethod
    def transform_wcs2ccs(coords_ws, inv_rotmat, translation):
        # print(np.shape(coords_ws))

        coords_cs = np.zeros(coords_ws.shape, dtype=coords_ws.dtype)
        for i in range(0, coords_ws.shape[1]):
            coords_cs[::, i] = np.dot(inv_rotmat, (coords_ws[::, i] - translation))

        return coords_cs

    ''' Affine transformation (vectors) from world space to camera space '''
    @staticmethod
    def transform_wc2cs_vectors(coords_ws, rotmat):
        coords_cs = np.zeros(coords_ws.shape, dtype=coords_ws.dtype)
        for i in range(0, coords_ws.shape[1]):
            coords_cs[::, i] = np.dot(rotmat, coords_ws[::, i])

        return coords_cs

    ''' Forth order spherical harmonics '''
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

    ''' Second Order Spherical Harmonics '''
    @staticmethod
    def spherical_harmonics(x, y, z, b):
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

    ''' Illumination Model '''
    def get_color(self, reflectance, normal, illumination):
        illumination = np.reshape(illumination, (3, 9), order='F')
        _sum = 0
        for i in range(0, 9):
            _sum = _sum + illumination[:, i] * self.spherical_harmonics(normal[0], normal[1], normal[2], i)

        color = np.multiply(reflectance, _sum)
        color = np.transpose(color)
        return color

    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    @staticmethod
    def normalize_v3(arr):
        lens = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2 + arr[:, 2] ** 2)
        arr[:, 0] /= lens
        arr[:, 1] /= lens
        arr[:, 2] /= lens
        return arr

    ''' Calculate Normals for each vertex '''
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

    ''' Image Formation - projected coordinates and color of each vertex '''
    def get_image_formation(self):
        ws_vertices = np.reshape(self.vertices, (3, int(self.vertices.size / 3)), order='F')
        reflectance = np.reshape(self.reflectance, (3, int(self.reflectance.size / 3)), order='F')

        rotmatSO3 = self.create_rot_mat(self.x['rotation'][0], self.x['rotation'][1], self.x['rotation'][2])
        inv_rotmat = np.linalg.inv(rotmatSO3)

        # Calculate color
        # world space normals
        ws_normals = self.calculate_normals(self.cells)

        # transform world space normals to camera space normals
        cs_normals = self.transform_wcs2ccs(ws_normals, inv_rotmat, self.x['translation'])

        # calculate color
        color = np.zeros(reflectance.shape, dtype=reflectance.dtype)
        for i in range(0, reflectance.shape[1]):
            color[:, i] = self.get_color(reflectance[:, i], cs_normals[:, i], self.x['illumination'])

        ''' Calculate projected coordinates '''
        cs_vertices = self.transform_wcs2ccs(ws_vertices, inv_rotmat, self.x['translation'])
        projected = self.projection(cs_vertices)

        formation = {
            "position": projected,
            "color": color
        }

        return formation

    ''' returns cells ordered deepest at top '''
    def calculate_cell_depth(self):

        vertices = np.reshape(self.vertices, (3, int(self.vertices.size / 3)), order='F')
        depth = np.zeros(self.cells.shape[1], dtype=self.cells.dtype)
        for i in range(0, self.cells.shape[1]):
            depth[i] = (vertices[2, self.cells[0, i]] + vertices[2, self.cells[1, i]] +
                        vertices[2, self.cells[2, i]]) / 3

        # arrange cells with deepest one first
        order = np.argsort(depth)
        cells_ordered = self.cells[:, order.astype(int)]

        # print(cells_ordered.shape)
        return cells_ordered
