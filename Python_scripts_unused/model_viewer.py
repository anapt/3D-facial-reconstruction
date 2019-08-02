import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def read_h5():
    model = h5py.File('/home/anapt/Documents/Thesis - data/data-raw/model2017-1_bfm_nomouth.h5', 'r+')
    return model


model = read_h5()


def read_scv():

    shape_pca = model['shape']['model']['pcaBasis'][()]
    shape_pca = shape_pca[1:len(shape_pca), 0:80]
    # print(shape_pca.shape)

    reflectance_pca = model['color']['model']['pcaBasis'][()]
    reflectance_pca = reflectance_pca[1:len(reflectance_pca), 0:80]
    # print(reflectance_pca.shape)

    expression_pca = model['expression']['model']['pcaBasis'][()]
    expression_pca = expression_pca[1:len(expression_pca), 0:64]
    # print(expression_pca.shape)

    average_shape = model['shape']['model']['mean'][()]
    # print(average_shape.shape)
    # print(type(average_shape))

    average_reflectance = model['color']['model']['mean'][()]
    # print(average_color.shape)                        # (159447,)
    # print(type(average_color))                        # <class 'numpy.ndarray'>

    return shape_pca, expression_pca, reflectance_pca, average_shape, average_reflectance


shape_pca, expression_pca, reflectance_pca, average_shape, average_reflectance = read_scv()
'''
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
'''

'''
plot_color = np.asarray(colorPCA['mean'])
plot_color = np.reshape(plot_color, (3, int(plot_color.size/3)), order='F')

plot_points = np.asarray(average_shape['points'])
'''
plot_color = np.asarray(average_reflectance)
plot_color = np.reshape(plot_color, (3, int(plot_color.size/3)), order='F')

plot_points = np.asarray(average_shape)
plot_points = np.reshape(plot_points, (3, int(plot_points.size/3)), order='F')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.scatter3D(plot_points[0, ], plot_points[1, ], plot_points[2, ], s=1, c=np.transpose(plot_color))

plt.show()
