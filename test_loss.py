from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import CollectBatchStats as batch_stats
from keras import backend as K


tf.compat.v1.enable_eager_execution()
BATCH_SIZE = 32
y_pred = tf.constant(1, shape=[BATCH_SIZE, 257], dtype=tf.float32)

shape = tf.slice(y_pred, begin=[0, 0], size=[BATCH_SIZE, 80], name='shape')
expression = tf.slice(y_pred, begin=[0, 80], size=[BATCH_SIZE, 64], name='expression')
reflectance = tf.slice(y_pred, begin=[0, 144], size=[BATCH_SIZE, 80], name='reflectance')
rotation = tf.slice(y_pred, begin=[0, 224], size=[BATCH_SIZE, 3], name='rotation')
translation = tf.slice(y_pred, begin=[0, 227], size=[BATCH_SIZE, 3], name='translation')
illumination = tf.slice(y_pred, begin=[0, 230], size=[BATCH_SIZE, 27], name='illumination')
print(shape)

shape2 = tf.math.square(shape, name='shape_squared')

reflectance2 = tf.math.multiply(tf.math.square(reflectance), 1.7*10e-3)
expression2 = tf.math.multiply(tf.math.square(expression), 0.8)

# print(tf.math.reduce_sum(shape2, axis=1))  # shape = (32, )

regularization_term = tf.math.reduce_sum(shape2, axis=1) + tf.math.reduce_sum(reflectance2, axis=1) + \
                      tf.math.reduce_sum(expression2, axis=1)


print(regularization_term)

regularization_term = tf.math.reduce_mean(regularization_term)  # shape = ()

print(regularization_term)

def regularization_term(shape, expression, reflectance):

    shape2 = tf.math.square(shape, name='shape_squared')
    reflectance2 = tf.math.multiply(tf.math.square(reflectance), 1.7 * 10e-3)
    expression2 = tf.math.multiply(tf.math.square(expression), 0.8)

    regularization_term = tf.math.reduce_sum(shape2, axis=1) + tf.math.reduce_sum(reflectance2, axis=1) + \
                          tf.math.reduce_sum(expression2, axis=1)

    regularization_term = tf.math.reduce_mean(regularization_term)  # shape = ()

    return regularization_term
