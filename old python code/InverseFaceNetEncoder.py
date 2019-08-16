from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from keras import backend as K
import SemanticCodeVector as scv

tf.compat.v1.enable_eager_execution()


class InverseFaceNetEncoder(object):

    def __init__(self):
        # Parameters
        self.IMG_SIZE = 240
        self.IMG_SHAPE = (self.IMG_SIZE, self.IMG_SIZE, 3)

        self.WEIGHT_DECAY = 0.001
        # self.WEIGHT_DECAY = 0.0000001
        self.BASE_LEARNING_RATE = 0.01

        self.BATCH_SIZE = 32
        self.BATCH_ITERATIONS = 75000

        self.SHUFFLE_BUFFER_SIZE = 20000

        # Parameters for Loss
        self.PATH = './DATASET/model2017-1_bfm_nomouth.h5'
        self.MAX_FEATURES = 500
        self.GOOD_MATCH_PERCENT = 0.15

        # Model
        self.model = self.build_model()
        # Print a model summary
        self.model.summary()

        # Loss Function
        self.loss_func = self.model_loss()

        # self.data = scv.SemanticCodeVector('./DATASET/model2017-1_bfm_nomouth.h5')
        # self.shape_sdev, self.reflectance_sdev, self.expression_sdev = self.data.get_parameters_dim_sdev()

    def build_model(self):
        """
         Create a Keras model

        :return: Keras.model()
        """

        # Create the base model from the pre-trained model XCEPTION
        # base_model = tf.keras.applications.xception.Xception(include_top=False,
        #                                                      weights='imagenet',
        #                                                      input_tensor=None,
        #                                                      input_shape=self.IMG_SHAPE,
        #                                                      pooling='avg')
        base_model = tf.keras.applications.vgg16.VGG16(include_top=False,
                                                       weights='imagenet',
                                                       input_shape=self.IMG_SHAPE,
                                                       pooling='avg')
        base_model.trainable = False
        base_model.summary()
        weights_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

        # Create global average pooling layer
        # global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        # global_average_layer = tf.keras.layers.GlobalMaxPooling2D()

        # Create prediction layer
        prediction_layer = tf.keras.layers.Dense(257, activation=None, use_bias=True,
                                                 kernel_initializer=weights_init, bias_initializer='zeros',
                                                 # kernel_regularizer=tf.keras.regularizers.l2(self.WEIGHT_DECAY)
                                                 )

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
            prediction_layer
        ])

        return model_o

    @staticmethod
    def model_space_parameter_loss(y):
        """
        Custom loss function that incorporates weights for each variable in the
        Semantic Code Vector

        :param y: Tensor, y_pred - y_true with shape (Batch_size, 257)
        :return: Float32, mean loss
        """
        # std_shape = tf.constant(self.shape_sdev, dtype=tf.float32)
        # std_shape = tf.compat.v1.reshape(std_shape, shape=(80,))
        # # shape_var = K.tile(std_shape, self.BATCH_SIZE)
        # shape_var = std_shape
        # # weight
        # shape_var = tf.math.scalar_mul(5000, shape_var, name='shape_var')
        #
        # std_expression = tf.constant(self.expression_sdev, dtype=tf.float32)
        # std_expression = tf.compat.v1.reshape(std_expression, shape=(64,))
        # # expression_var = K.tile(std_expression, self.BATCH_SIZE)
        # expression_var = std_expression
        # # weight
        # expression_var = tf.math.scalar_mul(5000, expression_var, name='expression_var')
        #
        # std_reflectance = tf.constant(self.reflectance_sdev, dtype=tf.float32)
        # std_reflectance = tf.compat.v1.reshape(std_reflectance, shape=(80,))
        # # reflectance_var = K.tile(std_reflectance, self.BATCH_SIZE)
        # reflectance_var = std_reflectance
        # # weight
        # reflectance_var = tf.math.scalar_mul(1000, reflectance_var, name='reflectance_var')

        # with tf.device('/device:GPU:1'):
        shape = tf.constant(125, shape=(1,), dtype=tf.float32)
        shape = K.tile(shape, 80)

        expression = tf.constant(1, shape=(1,), dtype=tf.float32)
        # expression2 = tf.constant(0, shape=(1,), dtype=tf.float32)
        expression = K.tile(expression, 64)
        # expression = tf.compat.v1.concat([expression, expression2], axis=0)

        reflectance = tf.constant(5500, shape=(1,), dtype=tf.float32)
        reflectance = K.tile(reflectance, 80)

        rotation = tf.constant(55, shape=(1,), dtype=tf.float32)
        rotation = K.tile(rotation, 3)

        translation = tf.constant(750, shape=(1,), dtype=tf.float32)
        translation = K.tile(translation, 3)

        illumination = tf.constant(3300, shape=(1,), dtype=tf.float32)
        illumination = K.tile(illumination, 27)

        sigma = tf.compat.v1.concat([shape, expression, reflectance, rotation, translation, illumination],
                                    axis=0)

        sigma = tf.linalg.tensor_diag(sigma)

        alpha = tf.linalg.matmul(sigma, y, transpose_b=True)

        beta = tf.linalg.matmul(alpha, alpha, transpose_a=True)

        loss = K.mean(beta, axis=-1)

        return loss

    def model_loss(self):
        """" Wrapper function which calculates auxiliary values for the complete loss function.
         Returns a *function* which calculates the complete loss given only the input and target output """

        # Model space parameter loss
        model_space_loss = self.model_space_parameter_loss

        def custom_loss(y_true, y_pred):
            # with tf.device('/device:GPU:1'):
            y = tf.math.subtract(y_pred, y_true, name='pred_minus_true')

            # Model Space Parameter Loss
            model_loss = model_space_loss(y)

            return model_loss

        return custom_loss

    def compile(self):
        """ Compiles the Keras model. Includes metrics to differentiate between the two main loss terms """
        self.model.compile(optimizer=tf.keras.optimizers.Adadelta(lr=self.BASE_LEARNING_RATE,
                                                                  rho=0.95, epsilon=None, decay=self.WEIGHT_DECAY),
                           loss=self.loss_func,
                           metrics=[tf.keras.losses.mean_squared_error, tf.keras.losses.mean_absolute_error])
        print('Model Compiled!')
