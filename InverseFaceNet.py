from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from keras import backend as K
tf.compat.v1.enable_eager_execution()


import numpy as np
import cv2
import SemanticCodeVector as scv
import ParametricMoDecoder as pmd
import LandmarkDetection as ld
import FaceCropper as fc
from patchImage import patch


class InverseFaceNetModel(object):

    def __init__(self):
        # Parameters
        self.IMG_SIZE = 240
        self.IMG_SHAPE = (self.IMG_SIZE, self.IMG_SIZE, 3)

        self.WEIGHT_DECAY = 0.001
        self.BASE_LEARNING_RATE = 0.01

        self.BATCH_SIZE = 20
        self.BATCH_ITERATIONS = 75000

        self.SHUFFLE_BUFFER_SIZE = 1000

        # Parameters for Loss
        self.PATH = './DATASET/model2017-1_bfm_nomouth.h5'
        self.MAX_FEATURES = 500
        self.GOOD_MATCH_PERCENT = 0.15
        self.photo_loss = 0
        self.reg_loss = 0

        # Model
        self.model = self.build_model()
        # Print a model summary
        self.model.summary()

        # Loss Function
        self.loss_func = self.model_loss()

    def build_model(self):
        """
         Create a Keras model

        :return: Keras.model()
        """

        # Create the base model from the pre-trained model XCEPTION
        base_model = tf.keras.applications.xception.Xception(include_top=False,
                                                             weights='imagenet',
                                                             input_tensor=None,
                                                             input_shape=self.IMG_SHAPE)
        base_model.trainable = False
        # base_model.summary()
        weights_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

        # Create global average pooling layer
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

        # Create prediction layer
        prediction_layer = tf.keras.layers.Dense(257, activation=None, use_bias=True,
                                                 kernel_initializer=weights_init, bias_initializer='zeros',
                                                 kernel_regularizer=tf.keras.regularizers.l2(self.WEIGHT_DECAY))

        # Code to check compatibility of dimensions
        # for image_batch, label_batch in keras_ds.take(1):
        #     pass
        # print('image batch shape ', image_batch.shape)
        # feature_batch = base_model(image_batch)
        # print('feature batch shape', feature_batch.shape)
        # feature_batch_average = global_average_layer(feature_batch)
        # print(feature_batch_average.shape)
        # prediction_batch = prediction_layer(feature_batch_average)
        # print(prediction_batch.shape)

        # print("Number of layers in the base model: ", len(base_model.layers))

        # Stack model layers
        model_o = tf.keras.Sequential([
            base_model,
            global_average_layer,
            prediction_layer
        ])

        return model_o

    def vector2dict(self, vector):
        """
        Transform vector to dictionary

        :param vector: vector with shape (257,)
        :return: dictionary of Semantic Code Vector
        """
        print("vector", vector)

        x = {
            "shape": vector[0:80, ],
            "expression": vector[80:144, ],
            "reflectance": vector[144:224, ],
            "rotation": vector[224:227, ],
            "translation": vector[227:230, ],
            "illumination": vector[230:257, ]
        }

        return x

    @staticmethod
    def statistical_regularization_term(x):
        weight_expression = 0.8
        weight_reflectance = 1.7e-3
        sr_term = K.sum(tf.map_fn(lambda t: t*t, x['shape']), axis=0) \
            + weight_expression * K.sum(tf.map_fn(lambda t: t*t, x['expression'])) \
            + weight_reflectance * K.sum(tf.map_fn(lambda t: t*t, x['reflectance']))

        # print("statistical reg error ", sr_term)
        return sr_term

    # def model_space_parameter_loss(self, x):

    def model_loss(self):
        """" Wrapper function which calculates auxiliary values for the complete loss function.
         Returns a *function* which calculates the complete loss given only the input and target output """

        # Model space parameter loss
        model_space_loss = self.model_space_parameter_loss
        # Regularization Loss
        reg_loss_func = self.statistical_regularization_term

        def custom_loss(y_true, y_pred):
            y_pred = K.transpose(y_pred)
            x = self.vector2dict(y_pred)

            y_true = K.transpose(y_true)

            tf.math.subtract(y_pred-y_true)

            print("x vector", x)


            # Regularization Loss
            reg_loss = reg_loss_func(x)
            # Photometric alignment loss
            # photo_loss = photo_loss_func(x, original_image)

            weight_photo = 1.92
            weight_reg = 2.9e-5

            # model_loss = weight_photo*photo_loss + weight_reg*reg_loss
            model_loss = weight_reg * reg_loss

            return model_loss

        return custom_loss

    def compile(self):
        """ Compiles the Keras model. Includes metrics to differentiate between the two main loss terms """
        self.model.compile(optimizer=tf.keras.optimizers.Adadelta(lr=self.BASE_LEARNING_RATE,
                                                                  rho=0.95, epsilon=None, decay=0.0),
                           loss=self.loss_func,
                           metrics=['mean_absolute_error'])
        print('Model Compiled!')
