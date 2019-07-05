import numpy as np


def normalize_v3(arr):
    # ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2)
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def calculate_normals(vertices, cells):
    # prepare data
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
    n = normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex,
    # so we need to normalize again afterwards.
    norm[cells[:, 0]] += n
    norm[cells[:, 1]] += n
    norm[cells[:, 2]] += n
    norm = normalize_v3(norm)

    return norm
