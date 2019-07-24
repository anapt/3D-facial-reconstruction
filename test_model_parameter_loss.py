import tensorflow as tf
import numpy as np
import keras.backend as K


def vector2dict(vector):
    """
    Transform vector to dictionary

    :param vector: vector with shape (257,)
    :return: dictionary of Semantic Code Vector
    """
    # print("vector", vector)

    x = {
        "shape": vector[0:80, ],
        "expression": vector[80:144, ],
        "reflectance": vector[144:224, ],
        "rotation": vector[224:227, ],
        "translation": vector[227:230, ],
        "illumination": vector[230:257, ]
    }

    return x


vector_path = ("./DATASET/semantic/x_%d.txt" % 1)
y_true = np.loadtxt(vector_path)

vector_path = ("./DATASET/semantic/x_%d.txt" % 0)
y_pred = np.loadtxt(vector_path)

x = vector2dict(y_pred)
y_pred = tf.constant(y_pred)
y_true = tf.constant(y_true)


y = tf.math.subtract(y_pred, y_true, name='pred_minus_true')

print(y)

y_transpose = K.transpose(y)

print(y_transpose)

# sigma = np.ones(shape=(257, 257))
var_shape = np.var(x['shape'])
# print(var_shape)
# var_shape = tf.compat.v1.numpy_function(np.var, [x['shape']], Tout=tf.float64)
var_shape = tf.constant(var_shape, dtype=tf.float64, shape=(1,))
shape_var = K.tile(var_shape, 80)

var_expression = np.var(x['expression'])
var_expression = tf.constant(var_expression, dtype=tf.float64, shape=(1,))
expression_var = K.tile(var_expression, 64)

var_reflectance = np.var(x['reflectance'])
var_reflectance = tf.constant(var_reflectance, dtype=tf.float64, shape=(1,))
reflectance_var = K.tile(var_reflectance, 80)

rotation = tf.constant(400, shape=(1,), dtype=tf.float64)
rotation = K.tile(rotation, 3)

illumination = tf.constant(20, shape=(1,), dtype=tf.float64)
illumination = K.tile(illumination, 27)

translation = tf.constant(5, shape=(1,), dtype=tf.float64)
translation = K.tile(translation, 3)

# diagonal = [shape_var, expression_var, reflectance_var, rotation, translation, illumination]

sigma = tf.compat.v1.concat([shape_var, expression_var, reflectance_var, rotation, translation, illumination], axis=0)

sigma = tf.linalg.tensor_diag(sigma)
print(sigma)

# print(tf.compat.v2.shape(var_shape))
# var_expression = tf.compat.v1.numpy_function(np.var, [x['expression']], Tout=tf.float32)
# var_reflectance = tf.compat.v1.numpy_function(np.var, [x['reflectance']], Tout=tf.float32)
#
# sigma = [50*var_shape, 50*var_expression, 100*var_reflectance]
# # sigma = tf.constant(sigma)
# # print(sigma)