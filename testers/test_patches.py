import numpy as np
import math as math
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# project classes
import ParametricMoDecoder as pmd
import SemanticCodeVector as scv

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

cells_ordered = decoder.calculate_cell_depth()

positionX = np.zeros((3, cells_ordered.shape[1]), dtype=formation['position'].dtype)
positionY = np.zeros((3, cells_ordered.shape[1]), dtype=formation['position'].dtype)
color = np.zeros((formation['color'].shape[0], cells_ordered.shape[1], 3), dtype=formation['color'].dtype)
# print(position.shape)
# print(formation['position'][:, cells_ordered[:, 1]])
for i in range(0, cells_ordered.shape[1]):
    positionX[:, i] = formation['position'][0, cells_ordered[:, i]]
    positionY[:, i] = formation['position'][1, cells_ordered[:, i]]
    color[:, i, :] = formation['color'][:, cells_ordered[:, i]]

fig, ax = plt.subplots()
print("here")
# print(position.shape)
# print(type(position))
# print(position)
# print(color)
patches = []
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-0.25, 2.75)

position = np.zeros((3, 2), dtype=positionY.dtype)
for i in range(0, cells_ordered.shape[1]):
    position[:, 0] = positionX[:, i]
    position[:, 1] = positionY[:, i]
    polygon = Polygon(position, True)
    polygon.set_fill(True)
    # col = color[:, i, :]
    # polygon.set_facecolor(col[:, 1])
    patches.append(polygon)

p = PatchCollection(patches, alpha=1)
# p.set_facecolor(color)
# colors = 100*np.random.rand(len(patches))
p.set_color(np.asarray(np.transpose(np.mean(color, axis=2))))
ax.add_collection(p)

plt.show()


# matlab code
# X = zeros(3,105694);
# Y = zeros(3,105694);
# C = ones(3,105694, 3);
#
# for i=1:1:105693
#     X(:, i) = projected(1, cells(:,i)+1);
#     Y(:, i) = projected(2, cells(:,i)+1);
#     C(:, i, :) = colors(:, cells(:,i)+1).';
# end
#
# patch(X, Y, C, 'edgecolor','none')
