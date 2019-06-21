import h5py
import numpy as np
import matplotlib.pyplot as plt

import math as math
import parametricMoDecoder as pmd
import semanticCodeVector as scv

# MODIFY TO path containing Basel Face model
path = '/home/anapt/Documents/Thesis - data/data-raw/model2017-1_bfm_nomouth.h5'
data = scv.SemanticCodeVector(path)
x = data.sample_vector()
cells = data.read_cells()

vertices = data.read_pca_bases()['average_shape']
print(vertices)
reflectance = data.read_pca_bases()['average_reflectance']


decoder = pmd.ParametricMoDecoder(vertices, reflectance)

vertices = np.reshape(vertices, (3, int(vertices.size / 3)), order='F')
reflectance = np.reshape(reflectance, (3, int(reflectance.size / 3)), order='F')

rotmatSO3 = decoder.create_rot_mat(x['rotation'][0], x['rotation'][1], x['rotation'][2])
inv_rotmat = np.linalg.inv(rotmatSO3)
print(vertices.shape[1])
ccs = decoder.transform_wcs2ccs(vertices, inv_rotmat, x['translation'])

projected = decoder.projection(ccs)

