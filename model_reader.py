import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


model = h5py.File('/home/anapt/Documents/Thesis - data/data-raw/model2017-1_bfm_nomouth.h5', 'r+')

shapePCA = model['shape']['model']
# ['mean', 'noiseVariance', 'pcaBasis', 'pcaVariance']
expressionPCA = model['expression']['model']
# ['mean', 'noiseVariance', 'pcaBasis', 'pcaVariance']
colorPCA = model['color']['model']
# ['mean', 'noiseVariance', 'pcaBasis', 'pcaVariance']

average_shape = model['shape']['representer']
# ['cells', 'points']
average_color = model['color']['representer']
# ['cells', 'colorspace', 'points']

plot_color = np.asarray(colorPCA['mean'])
plot_color = np.reshape(plot_color, (3, int(plot_color.size/3)), order='F')

plot_points = np.asarray(average_shape['points'])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.scatter3D(plot_points[0, ], plot_points[1, ], plot_points[2, ], s=1, c=np.transpose(plot_color))

plt.show()
