from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import CollectBatchStats as batch_stats
from keras import backend as K
import SemanticCodeVector as scv

data = scv.SemanticCodeVector('./DATASET/model2017-1_bfm_nomouth.h5')
bases = data.read_pca_bases()
avg_shape = bases['average_shape']
shape_pca = bases['shape_pca']
reflectance_pca = bases['reflectance_pca']
expression_pca = bases['expression_pca']
avg_reflectance = bases['average_reflectance']

tf.compat.v1.enable_eager_execution()
BATCH_SIZE = 32
y_true = np.loadtxt('./DATASET/semantic/over/x_{:06}.txt'.format(0))
y_pred = np.loadtxt('./DATASET/semantic/over/x_{:06}.txt'.format(0))
y_pred = tf.constant(y_true, shape=[1, 257], dtype=tf.float32)
# y_true = tf.constant(y_true, shape=[BATCH_SIZE, 257], dtype=tf.float32)

# shape = tf.slice(y_pred, begin=[0, 0], size=[BATCH_SIZE, 80], name='shape')
# expression = tf.slice(y_pred, begin=[0, 80], size=[BATCH_SIZE, 64], name='expression')
# reflectance = tf.slice(y_pred, begin=[0, 144], size=[BATCH_SIZE, 80], name='reflectance')
# rotation = tf.slice(y_pred, begin=[0, 224], size=[BATCH_SIZE, 3], name='rotation')
# translation = tf.slice(y_pred, begin=[0, 227], size=[BATCH_SIZE, 3], name='translation')
# illumination = tf.slice(y_pred, begin=[0, 230], size=[BATCH_SIZE, 27], name='illumination')
# print(shape)
shape = tf.slice(y_pred, begin=[0, 0], size=[1, 80], name='shape')
expression = tf.slice(y_pred, begin=[0, 80], size=[1, 64], name='expression')
reflectance = tf.slice(y_pred, begin=[0, 144], size=[1, 80], name='reflectance')
#
# shape2 = tf.math.square(shape, name='shape_squared')
#
# # reflectance2 = tf.math.multiply(tf.math.square(reflectance), 1.7*10e-3)
# # expression2 = tf.math.multiply(tf.math.square(expression), 0.8)
# reflectance2 = tf.math.multiply(tf.math.square(reflectance), 35)
# expression2 = tf.math.multiply(tf.math.square(expression),  1.7*10e-3)
# print(tf.math.reduce_sum(reflectance2))
# print(tf.math.reduce_sum(expression2))
# print(tf.math.reduce_sum(shape2))
# # print(tf.math.reduce_sum(shape2, axis=1))  # shape = (32, )
#
# regularization_term = tf.math.reduce_sum(shape2, axis=1) + tf.math.reduce_sum(reflectance2, axis=1) + \
#                       tf.math.reduce_sum(expression2, axis=1)
#
#
# print(regularization_term)
#
# regularization_term = tf.math.reduce_mean(regularization_term)  # shape = ()
#
# print(regularization_term)

# input = tf.constant(1, shape=[BATCH_SIZE, 240, 240, 3], dtype=tf.float32)
#
# shape = tf.slice(y_pred, begin=[0, 0], size=[BATCH_SIZE, 80], name='shape')
# expression = tf.slice(y_pred, begin=[0, 80], size=[BATCH_SIZE, 64], name='expression')
# reflectance = tf.slice(y_pred, begin=[0, 144], size=[BATCH_SIZE, 80], name='reflectance')

avg_shape = tf.constant(avg_shape, shape=[159447, 1], dtype=tf.float32)
shape_pca = tf.constant(shape_pca, dtype=tf.float32)
expression_pca = tf.constant(expression_pca, dtype=tf.float32)
avg_reflectance = tf.constant(avg_reflectance, shape=[159447, 1], dtype=tf.float32)
reflectance_pca = tf.constant(reflectance_pca, dtype=tf.float32)

# shape = 159447, BATCH_SIZE or 3N, BATCH_SIZE
alpha = tf.math.add(avg_shape, tf.linalg.matmul(shape_pca, shape, transpose_b=True))
vertices_pred = tf.math.add(alpha, tf.linalg.matmul(expression_pca, expression, transpose_b=True))
skin_ref_pred = tf.math.add(avg_reflectance, tf.linalg.matmul(reflectance_pca, reflectance, transpose_b=True))

print(tf.math.reduce_sum(vertices_pred))
print(tf.math.reduce_sum(skin_ref_pred))

print(tf.math.multiply(tf.math.reduce_sum(vertices_pred), 2.99*10e-2))
print(tf.math.multiply(tf.math.reduce_sum(skin_ref_pred), 1))
#
# shape = tf.slice(y_true, begin=[0, 0], size=[BATCH_SIZE, 80], name='shape')
# expression = tf.slice(y_true, begin=[0, 80], size=[BATCH_SIZE, 64], name='expression')
# reflectance = tf.slice(y_true, begin=[0, 144], size=[BATCH_SIZE, 80], name='reflectance')
#
# # shape = 159447, BATCH_SIZE or 3N, BATCH_SIZE
# alpha = tf.math.add(avg_shape, tf.linalg.matmul(shape_pca, shape, transpose_b=True))
# vertices_true = tf.math.add(alpha, tf.linalg.matmul(expression_pca, expression, transpose_b=True))
# skin_ref_true = tf.math.add(avg_reflectance, tf.linalg.matmul(reflectance_pca, reflectance, transpose_b=True))
#
# vertices_dist = tf.linalg.norm(vertices_pred - vertices_true, ord='euclidean', axis=0)
# print(vertices_dist)
#
# reflectance_dist = tf.linalg.norm(skin_ref_pred - skin_ref_true, ord='euclidean', axis=0)
# print(reflectance_dist)
#
loss_vertices = tf.math.reduce_mean(tf.math.multiply(vertices_dist, 2.99*10e-2))
loss_reflectance = tf.math.reduce_mean(reflectance_dist)
#
# print(loss_reflectance, loss_vertices)


