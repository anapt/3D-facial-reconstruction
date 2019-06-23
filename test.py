import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

import math as math
import parametricMoDecoder as pmd
import semanticCodeVector as scv

''' SEMANTIC CODE VECTOR '''
# MODIFY TO path containing Basel Face model
path = '/home/anapt/Documents/Thesis - data/data-raw/model2017-1_bfm_nomouth.h5'
data = scv.SemanticCodeVector(path)
x = data.sample_vector()
cells = data.read_cells()

vertices = data.read_pca_bases()['average_shape']
# reflectance = data.read_pca_bases()['average_reflectance']

reflectance = data.calculate_reflectance(x)

''' PARAMETRIC MODEL BASED DECODER '''
decoder = pmd.ParametricMoDecoder(vertices, reflectance)
# vertices and reflectance reshape in [3, 3N]
vertices = np.reshape(vertices, (3, int(vertices.size / 3)), order='F')
reflectance = np.reshape(reflectance, (3, int(reflectance.size / 3)), order='F')
np.savetxt('reflectance.txt', reflectance)
# rotation matrix -> for testing no rotation
rotmatSO3 = decoder.create_rot_mat(x['rotation'][0], x['rotation'][1], x['rotation'][2])
inv_rotmat = np.linalg.inv(rotmatSO3)

# from WCS to CCS
ccs = decoder.transform_wcs2ccs(vertices, inv_rotmat, x['translation'])
# from CCS to SCREEN SPACE
projected = decoder.projection(ccs)
# TODO maybe change to transpose for consistency
projected = np.transpose(projected)
print(projected.shape)
# 53149 2

# plot no color
# plt.scatter(projected[:, 0], projected[:, 1], alpha=0.01)
# plt.show()

# calculate normals
normals = decoder.calculate_normals(cells)
print(normals.shape)

# calculate color
color = np.zeros(reflectance.shape, dtype=reflectance.dtype)
print(color.shape)

np.savetxt('projected.txt', projected)

for i in range(0, reflectance.shape[1]):
    color[:, i] = decoder.get_color(reflectance[:, i], normals[:, i], x['illumination'])

print(color.shape)
np.savetxt('color.txt', color)
# plt.scatter(projected[0, ], projected[1, ], c=color, alpha=0.01)
# plt.show()

''' NORMALS PLOT '''
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter(normals[0, ], normals[1, ], normals[2, ], s=1)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# np.savetxt('normals.txt', normals)


plt.show()
