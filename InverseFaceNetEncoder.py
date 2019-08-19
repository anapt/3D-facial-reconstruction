from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from keras import backend as K
from FaceNet3D import FaceNet3D as Helpers
from SemanticCodeVector import SemanticCodeVector

tf.compat.v1.enable_eager_execution()


class InverseFaceNetEncoder(Helpers):

    def __init__(self):
        super().__init__()

        # Model
        self.model = self.build_model()
        # Print a model summary
        self.model.summary()

        # Loss Function
        self.loss_func = self.model_loss()

        self.data = SemanticCodeVector()
        self.shape_std, self.color_std, self.expression_std = self.data.get_bases_std()

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
        # TODO ADD CUSTOM INITIALIZER
        # weights_init = np.zeros((self.scv_length,))
        # weights_init[self.scv_length - self.translation_dim] = 2.5
        # weights_init[self.scv_length - self.translation_dim + 1] = 2.5
        # weights_init[self.scv_length - self.translation_dim + 2] = -50
        #
        # weights_init = tf.constant(weights_init, shape=(self.scv_length,), dtype=tf.float32)
        weights_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

        # Create global average pooling layer
        # global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        # global_average_layer = tf.keras.layers.GlobalMaxPooling2D()

        # Create prediction layer
        prediction_layer = tf.keras.layers.Dense(self.scv_length, activation=None, use_bias=True,
                                                 kernel_initializer=weights_init, bias_initializer='zeros'
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

    def model_space_parameter_loss(self, y):
        """
        Custom loss function that incorporates weights for each variable in the
        Semantic Code Vector

        :param y: Tensor, y_pred - y_true with shape (Batch_size, 257)
        :return: Float32, mean loss
        """
        # std_shape = tf.constant(self.shape_std, dtype=tf.float32)
        # std_shape = tf.compat.v1.reshape(std_shape, shape=(self.shape_dim,))
        # # std_shape = K.tile(std_shape, self.BATCH_SIZE)
        #
        # # # weight
        # shape = tf.math.scalar_mul(1.4, std_shape, name='shape_std')
        #
        # std_expression = tf.constant(self.expression_std, dtype=tf.float32)
        # std_expression = tf.compat.v1.reshape(std_expression, shape=(self.expression_dim,))
        # # std_expression = K.tile(std_expression, self.BATCH_SIZE)
        #
        # # # weight
        # expression = tf.math.scalar_mul(2.4, std_expression, name='expression_std')
        #
        # std_color = tf.constant(self.color_std, dtype=tf.float32)
        # std_color = tf.compat.v1.reshape(std_color, shape=(self.color_dim,))
        # # std_color = K.tile(std_color, self.BATCH_SIZE)
        #
        # # weight
        # color = tf.math.scalar_mul(50, std_color, name='shape_std')

        # with tf.device('/device:GPU:1'):
        shape = tf.constant(15, shape=(1,), dtype=tf.float32)
        shape = K.tile(shape, self.shape_dim)

        expression = tf.constant(15, shape=(1,), dtype=tf.float32)
        # expression2 = tf.constant(0, shape=(1,), dtype=tf.float32)
        expression = K.tile(expression, self.expression_dim)

        color = tf.constant(10, shape=(1,), dtype=tf.float32)
        color = K.tile(color, self.color_dim)

        rotation = tf.constant(500, shape=(1,), dtype=tf.float32)
        rotation = K.tile(rotation, self.rotation_dim)

        sigma = tf.compat.v1.concat([shape, expression, color, rotation],
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
                           # loss=tf.keras.losses.mean_squared_error,
                           metrics=[tf.keras.losses.mean_squared_error, tf.keras.losses.mean_absolute_error])
        print('Model Compiled!')
