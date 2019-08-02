from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import keras
from keras import backend as K
import numpy as np
from loadDataset import load_dataset
tf.compat.v1.enable_eager_execution()
# sess = tf.compat.v1.Session()
# sess.as_default()

# PARAMETERS
IMG_SHAPE = (240, 240, 3)
WEIGHT_DECAY = 0.001
BASE_LEARNING_RATE = 0.01

BATCH_SIZE = 20
BATCH_ITERATIONS = 75000

SHUFFLE_BUFFER_SIZE = 1000

# Parameters for Loss
PATH = './DATASET/model2017-1_bfm_nomouth.h5'
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


# Create the base model from the pre-trained model XCEPTION
base_model = tf.keras.applications.xception.Xception(include_top=False,
                                                     weights='imagenet',
                                                     input_tensor=None,
                                                     input_shape=IMG_SHAPE)
base_model.trainable = False
# base_model.summary()
weights_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

# Create global average pooling layer
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

# Create prediction layer
prediction_layer = tf.keras.layers.Dense(257, activation=None, use_bias=True,
                                         kernel_initializer=weights_init, bias_initializer='zeros',
                                         kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY))

model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])




# def model_loss():
#     """" Wrapper function which calculates auxiliary values for the complete loss function.
#      Returns a *function* which calculates the complete loss given only the input and target output """
#     print(tf.executing_eagerly())
#     # Photometric alignment Loss
#     # photo_loss_func = dense_photometric_alignment
#     # # Regularization Loss
#     reg_loss_func = statistical_regularization_term
#     print("here")
#     original_image = model.input
#     print("model input", original_image)
#
#     idx0 = tf.shape(original_image)[0]
#     print(idx0)
#     original_image = tf.cond(tf.equal(tf.size(idx0), 0),
#                              lambda: np.zeros(shape=(240, 240, 3), dtype=np.float32),
#                              lambda: model.input)
#     print("after condition check", original_image)
#     print(tf.executing_eagerly())
#     original_image = original_image.numpy()
#     print(original_image)
#     print("end")
#     # is_empty = tf.equal(tf.size(original_image), 0)
#     # if is_empty:
#     #     print("none")
#     #     original_image = np.zeros(shape=(240, 240, 3))
#     original_image = K.eval(original_image)
#     print("original image type", type(original_image))
#
#     def custom_loss(y_true, y_pred):
#         tf.compat.v1.enable_eager_execution()
#         print(tf.executing_eagerly())
#
#         original_image = model.input
#         original_image = tf.compat.v1.squeeze(original_image, 0)
#         # original_image = to_numpy(original_image)
#         print("type original image", type(original_image))
#
#         print(type(original_image))
#
#         # Regularization Loss
#         reg_loss = reg_loss_func(y_pred)
#         # Photometric alignment loss
#         print("photo loss")
#         # photo_loss = photo_loss_func(x, original_image)
#
#         weight_photo = 1.92
#         weight_reg = 2.9e-5
#
#         # model_loss = weight_photo * photo_loss + weight_reg * reg_loss
#         model_loss = weight_reg * reg_loss
#
#         return model_loss
#
#     return custom_loss


""" Compiles the Keras model. Includes metrics to differentiate between 
the two main loss terms """
model.compile(optimizer=tf.keras.optimizers.Adadelta(lr=BASE_LEARNING_RATE,
                                                     rho=0.95, epsilon=None, decay=0.0),
              loss=['mean_squared_error'],
              metrics=['mean_absolute_error'])
print('Model Compiled!')

print(tf.executing_eagerly())

graph_1 = tf.Graph()
with graph_1.as_default():
    var = tf.Variable(100, name='var')
    original_image = tf.Tensor(model.input)
    init_graph_1 = tf.compat.v1.global_variables_initializer()


with tf.compat.v1.Session(graph=graph_1) as sess_1:
    sess_1.run(init_graph_1)
    print(sess_1.run(original_image))  ## output 100

# Parameters
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 20
SHUFFLE_BUFFER_SIZE = 1000

checkpoint_path = "./DATASET/training/cp-{epoch:04d}.ckpt"
checkpoint_dir = "./DATASET/training/"

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=5)

keras_ds = load_dataset()
keras_ds = keras_ds.shuffle(SHUFFLE_BUFFER_SIZE).repeat().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

steps_per_epoch = tf.math.ceil(SHUFFLE_BUFFER_SIZE / BATCH_SIZE).numpy()
print("Training with %d steps per epoch" % steps_per_epoch)

model.fit(keras_ds, epochs=10, steps_per_epoch=steps_per_epoch, callbacks=[cp_callback])
