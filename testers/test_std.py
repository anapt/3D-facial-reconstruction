import numpy as np
import tensorflow as tf
import keras.backend as K
tf.compat.v1.enable_eager_execution()
import SemanticCodeVector as scv
data = scv.SemanticCodeVector('./DATASET/model2017-1_bfm_nomouth.h5')


shape_sdev, reflectance_sdev, expression_sdev = data.get_parameters_dim_sdev()


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


vector_path = ("./DATASET/semantic/x_%d.txt" % 3)
y_true = np.loadtxt(vector_path)

vector_path = ("./DATASET/semantic/x_%d.txt" % 2)
y_pred = np.loadtxt(vector_path)
#
#
# x = vector2dict(y_true)
y = tf.math.subtract(y_pred, y_true, name='pred_minus_true')
# print(np.var(x['shape'], axis=0))
print(y.numpy())


std_shape = tf.constant(shape_sdev, dtype=tf.float32)
std_shape = tf.compat.v1.reshape(std_shape, shape=(80,))
shape_var = std_shape
# weight
shape_var = tf.math.scalar_mul(5000, shape_var, name='shape_var')

std_expression = tf.constant(expression_sdev, dtype=tf.float32)
std_expression = tf.compat.v1.reshape(std_expression, shape=(64,))
# expression_var = K.tile(std_expression, self.BATCH_SIZE)
expression_var = std_expression
# weight
expression_var = tf.math.scalar_mul(5000, expression_var, name='expression_var')

std_reflectance = tf.constant(reflectance_sdev, dtype=tf.float32)
std_reflectance = tf.compat.v1.reshape(std_reflectance, shape=(80,))
# reflectance_var = K.tile(std_reflectance, self.BATCH_SIZE)
reflectance_var = std_reflectance
# weight
reflectance_var = tf.math.scalar_mul(1000, reflectance_var, name='reflectance_var')
print("reflectance", reflectance_var.numpy())
rotation = tf.constant(4, shape=(1,), dtype=tf.float32)
rotation = K.tile(rotation, 3)

illumination = tf.constant(3, shape=(1,), dtype=tf.float32)
illumination = K.tile(illumination, 27)

translation = tf.constant(1, shape=(1,), dtype=tf.float32)
translation = K.tile(translation, 3)

sigma = tf.compat.v1.concat([shape_var, expression_var, reflectance_var, rotation, translation, illumination],
                                    axis=0)

sigma = tf.linalg.tensor_diag(sigma)

print(shape_var.numpy())
loss = tf.linalg.tensordot(y, y, axes=1)
print(loss.numpy())
