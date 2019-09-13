from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from keras import backend as K
import numpy as np
from FaceNet3D import FaceNet3D as Helpers
from SemanticCodeVector import SemanticCodeVector

tf.compat.v1.enable_eager_execution()


class InverseFaceNetEncoder(Helpers):

    def __init__(self):
        """
        Class initializer
        """
        super().__init__()

        # Model
        self.model = self.build_model()
        # Print a model summary
        # self.model.summary()

        # Custom Loss Function
        self.loss_func = self.model_loss()

        self.data = SemanticCodeVector()
        self.shape_std, self.color_std, self.expression_std = self.data.get_bases_std()
        self.scale_shape = self.shape_dim / np.sum(self.shape_std)
        self.scale_color = self.color_dim / np.sum(self.color_std)
        self.scale_expression = self.expression_dim / np.sum(self.expression_std)

        self.early = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=2, verbose=1,
                                                      mode='min', baseline=None, restore_best_weights=True)

    def build_model(self):
        """
         Create a Keras model

        :return: Keras.model()
        """
        # base_model = tf.keras.applications.resnet50.ResNet50(include_top=False,
        #                                                      weights='imagenet',
        #                                                      input_tensor=None,
        #                                                      # input_shape=self.IMG_SHAPE,
        #                                                      pooling='avg')
        base_model = tf.keras.applications.xception.Xception(include_top=False,
                                                             weights='imagenet',
                                                             input_tensor=None,
                                                             input_shape=self.IMG_SHAPE,
                                                             pooling='avg')

        base_model.trainable = True
        base_model.summary()

        weights_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

        # Create prediction layer
        prediction_layer = tf.keras.layers.Dense(self.scv_length, activation=None, use_bias=True,
                                                 kernel_initializer=weights_init, bias_initializer='zeros')

        # Stack model layers
        model_o = tf.keras.Sequential([
            base_model,
            prediction_layer
        ])
        """
        Model: "sequential"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        == == == == == == == == == == == == == == == == == == == == == ==
        resnet50(Model)             (None, 2048)                23587712
        _________________________________________________________________
        dense(Dense)                (None, 231)                 473319
        == == == == == == == == == == == == == == == == == == == == == == 
        Total params: 24,061,031
        Trainable params: 24,007,911
        Non - trainable params: 53,120 (from resnet50)
        _________________________________________________________________
        """
        return model_o

    def model_space_parameter_loss(self, y):
        """
        Custom loss function that incorporates weights for each variable in the
        Semantic Code Vector

        :param y: Tensor, y_pred - y_true with shape (Batch_size, 257)
        :return: Float32, mean loss
        """
        std_shape = tf.constant(self.shape_std, dtype=tf.float32)
        std_shape = tf.compat.v1.reshape(std_shape, shape=(self.shape_dim,))
        # weight
        shape = tf.math.scalar_mul(self.scale_shape, std_shape, name='shape_std')

        std_expression = tf.constant(self.expression_std, dtype=tf.float32)
        std_expression = tf.compat.v1.reshape(std_expression, shape=(self.expression_dim,))

        # weight
        expression = tf.math.scalar_mul(self.scale_expression, std_expression, name='expression_std')

        std_color = tf.constant(self.color_std, dtype=tf.float32)
        std_color = tf.compat.v1.reshape(std_color, shape=(self.color_dim,))

        # weight
        color = tf.math.scalar_mul(self.scale_color, std_color, name='color_std')

        rotation = tf.constant(1, shape=(1,), dtype=tf.float32)
        rotation = K.tile(rotation, self.rotation_dim)

        sigma = tf.compat.v1.concat([shape, expression, color, rotation],
                                    axis=0)

        alpha = tf.math.multiply(sigma, y)

        beta = K.mean(alpha)

        return beta

    def model_loss(self):
        """
        Wrapper function which calculates auxiliary values for the complete loss function.
         Returns a *function* which calculates the complete loss given only the input and target output
        """

        # Model space parameter loss
        model_space_loss = self.model_space_parameter_loss

        def custom_loss(y_true, y_pred):
            # with tf.device('/device:GPU:1'):
            y = K.square(y_pred - y_true)

            # Model Space Parameter Loss
            # with tf.device('/device:CPU:0'):
            model_loss = model_space_loss(y)

            return model_loss

        return custom_loss

    def compile(self):
        """
        Compiles the Keras model. Includes metrics to differentiate between the two main loss terms
        """
        self.model.compile(optimizer=tf.keras.optimizers.Adadelta(lr=self.BASE_LEARNING_RATE,
                                                                  rho=0.95, epsilon=None, decay=self.WEIGHT_DECAY),
                           loss=self.loss_func,
                           metrics=[tf.keras.losses.mean_squared_error, tf.keras.losses.mean_absolute_error])
        print('Model Compiled!')
