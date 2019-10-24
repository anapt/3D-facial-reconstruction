from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from keras import backend as K
import numpy as np
from FaceNet3D import FaceNet3D as Helpers
from SemanticCodeVector import SemanticCodeVector

tf.compat.v1.enable_eager_execution()


class InverseFaceNet(Helpers):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    def __init__(self):
        """
        Class initializer
        """
        super().__init__()

        # Model
        self.model = self.build_model()
        # Print a model summary
        # self.model.summary()

        # Loss Function
        self.loss_func = self.model_loss()

        self.data = SemanticCodeVector()
        self.bases = self.data.read_pca_bases()
        self.shape_pca, self.expression_pca, self.color_pca, self.avg_shape, self.avg_reflectance = \
            self.bases_dict_2_vectors()

    def bases_dict_2_vectors(self):

        avg_shape = self.bases['average_shape']
        shape_pca = self.bases['shape_pca']
        color_pca = self.bases['color_pca']
        expression_pca = self.bases['expression_pca']
        avg_reflectance = self.bases['average_reflectance']

        return shape_pca, expression_pca, color_pca, avg_shape, avg_reflectance

    def build_model(self):
        """
         Create a Keras model

        :return: Keras.model()
        """
        base_model = tf.keras.applications.resnet50.ResNet50(include_top=False,
                                                             weights='imagenet',
                                                             input_tensor=None,
                                                             # input_shape=self.IMG_SHAPE,
                                                             pooling='avg')

        base_model.trainable = True
        # base_model.summary()

        weights_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

        # Create prediction layer
        prediction_layer = tf.keras.layers.Dense(self.scv_length, activation=None, use_bias=True,
                                                 kernel_initializer=weights_init, bias_initializer='zeros')

        # Stack model layers
        model_o = tf.keras.Sequential([
            base_model,
            prediction_layer
        ])

        # load trained weights from bootstrapping
        latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        latest = './DATASET/training/cp-0090.ckpt'
        model_o.load_weights(latest)

        return model_o

    def regularization_term(self, y_pred):
        """
        Custom loss
        Regularization term : forces shape, expression and reflectances to be close to that of the
        average face

        :param y_pred: network output : Tensor with shape (BATCH_SIZE, 257)
        :return: regularization loss
        """
        shape = tf.slice(y_pred, begin=[0, 0], size=[self.BATCH_SIZE, self.shape_dim], name='shape')
        expression = tf.slice(y_pred, begin=[0, self.shape_dim], size=[self.BATCH_SIZE, self.expression_dim], name='expression')
        color = tf.slice(y_pred, begin=[0, self.shape_dim + self.expression_dim], size=[self.BATCH_SIZE, self.color_dim], name='color')

        shape2 = tf.math.multiply(tf.math.square(shape, name='shape_squared'), 0.9)
        reflectance2 = tf.math.multiply(tf.math.square(color), 32)
        expression2 = tf.math.multiply(tf.math.square(expression), 1.6 * 10e-3)

        regularization_term = tf.math.reduce_sum(shape2, axis=1) + tf.math.reduce_sum(reflectance2, axis=1) + \
            tf.math.reduce_sum(expression2, axis=1)

        regularization_term = tf.math.reduce_mean(regularization_term)  # shape = ()

        return regularization_term

    def photometric_term(self, y_true, y_pred):
        """
        Custom loss
        Regularization term : forces shape, expression and reflectances to be close to that of the
        average face

        :param y_true: ground truth semantic code vector : Tensor with shape (BATCH_SIZE, 257)
        :param y_pred: network output : Tensor with shape (BATCH_SIZE, 257)
        :return: photometric alignment loss
        """
        shape = tf.slice(y_pred, begin=[0, 0], size=[self.BATCH_SIZE, self.shape_dim], name='shape')
        expression = tf.slice(y_pred, begin=[0, self.shape_dim], size=[self.BATCH_SIZE, self.expression_dim],
                              name='expression')
        color = tf.slice(y_pred, begin=[0, self.shape_dim + self.expression_dim],
                         size=[self.BATCH_SIZE, self.color_dim], name='color')

        avg_shape = tf.constant(self.avg_shape, shape=[self.num_of_vertices, 1], dtype=tf.float32)
        shape_pca = tf.constant(self.shape_pca, dtype=tf.float32)
        expression_pca = tf.constant(self.expression_pca, dtype=tf.float32)
        avg_reflectance = tf.constant(self.avg_reflectance, shape=[self.num_of_vertices, 1], dtype=tf.float32)
        color_pca = tf.constant(self.color_pca, dtype=tf.float32)

        # shape = 159447, BATCH_SIZE or 3N, BATCH_SIZE
        alpha = tf.math.add(avg_shape, tf.linalg.matmul(shape_pca, shape, transpose_b=True))
        vertices_pred = tf.math.add(alpha, tf.linalg.matmul(expression_pca, expression, transpose_b=True))
        skin_ref_pred = tf.math.add(avg_reflectance, tf.linalg.matmul(color_pca, color, transpose_b=True))

        shape = tf.slice(y_true, begin=[0, 0], size=[self.BATCH_SIZE, self.shape_dim], name='shape')
        expression = tf.slice(y_true, begin=[0, self.shape_dim], size=[self.BATCH_SIZE, self.expression_dim],
                              name='expression')
        color = tf.slice(y_true, begin=[0, self.shape_dim + self.expression_dim],
                         size=[self.BATCH_SIZE, self.color_dim], name='color')

        # shape = 159447, BATCH_SIZE or 3N, BATCH_SIZE
        alpha = tf.math.add(avg_shape, tf.linalg.matmul(shape_pca, shape, transpose_b=True))
        vertices_true = tf.math.add(alpha, tf.linalg.matmul(expression_pca, expression, transpose_b=True))
        skin_ref_true = tf.math.add(avg_reflectance, tf.linalg.matmul(color_pca, color, transpose_b=True))

        vertices_dist = tf.linalg.norm(vertices_pred - vertices_true, ord=2, axis=0)
        reflectance_dist = tf.linalg.norm(skin_ref_pred - skin_ref_true, ord=2, axis=0)

        loss_vertices = tf.math.reduce_mean(vertices_dist)
        loss_reflectance = tf.math.reduce_mean(reflectance_dist)

        weight_vertices = 1
        weight_reflectance = 1

        photo_term = weight_vertices * loss_vertices + weight_reflectance * loss_reflectance

        return photo_term

    def model_loss(self):
        """" Wrapper function which calculates auxiliary values for the complete loss function.
         Returns a *function* which calculates the complete loss given only the input and target output """

        # Model space parameter loss
        regularization = self.regularization_term
        photometric = self.photometric_term
        # print(self.model.input)

        def custom_loss(y_true, y_pred):

            weight_reg = 0.06
            weight_photo = 1

            # Model Loss Layer
            # reg_term = regularization(y_pred)
            photo_term = photometric(y_true, y_pred)

            model_loss = weight_photo * photo_term

            return model_loss

        return custom_loss

    def compile(self):
        """ Compiles the Keras model. Includes metrics to differentiate between the two main loss terms """
        self.model.compile(optimizer=tf.keras.optimizers.Adadelta(lr=self.BASE_LEARNING_RATE,
                                                                  rho=0.95, epsilon=None, decay=self.WEIGHT_DECAY),
                           loss=self.loss_func,
                           metrics=[tf.keras.losses.mean_squared_error, tf.keras.losses.mean_absolute_error])
        print('Model Compiled!')
