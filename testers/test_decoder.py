import h5py
import numpy as np
import matplotlib.pyplot as plt

import ParametricMoDecoder as pmd
import SemanticCodeVector as scv

''' SEMANTIC CODE VECTOR '''
# MODIFY TO path containing Basel Face model
path = './DATASET/model2017-1_bfm_nomouth.h5'
data = scv.SemanticCodeVector(path)
x = data.sample_vector()
cells = data.read_cells()

vertices = data.read_pca_bases()['average_shape']
reflectance = data.read_pca_bases()['average_reflectance']

reflectance = data.calculate_reflectance(x)

''' PARAMETRIC MODEL BASED DECODER '''
decoder = pmd.ParametricMoDecoder(vertices, reflectance, x, cells)

formation = decoder.get_image_formation()
decoder.create_rot_mat(10, 10 , 10)
# np.savetxt('color.txt', formation['color'])
# np.savetxt('position.txt', formation['position'])

cells_ordered = decoder.calculate_cell_depth()
# np.savetxt('cells.txt', cells_ordered)
# np.savetxt('color.txt', color_ordered)

# plt.scatter(formation['position'][0, :], formation['position'][1, :], s=1)
# plt.show()



