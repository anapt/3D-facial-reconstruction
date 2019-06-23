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
reflectance = data.read_pca_bases()['average_reflectance']

reflectance = data.calculate_reflectance(x)

''' PARAMETRIC MODEL BASED DECODER '''
decoder = pmd.ParametricMoDecoder(vertices, reflectance, x, cells)

formation = decoder.get_image_formation()
np.savetxt('color.txt', formation['color'])
np.savetxt('position.txt', formation['position'])
