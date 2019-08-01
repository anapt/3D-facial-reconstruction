from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import CollectBatchStats as batch_stats
from keras import backend as K


tf.compat.v1.enable_eager_execution()


class InverseFaceNet(object):

    def __init__(self):
        # Parameters
        self.IMG_SIZE = 240
        self.IMG_SHAPE = (self.IMG_SIZE, self.IMG_SIZE, 3)

        # self.WEIGHT_DECAY = 0.001
        self.WEIGHT_DECAY = 0.0000001
        self.BASE_LEARNING_RATE = 0.1

        self.BATCH_SIZE = 1
        self.BATCH_ITERATIONS = 75000

        self.SHUFFLE_BUFFER_SIZE = 15

        # Parameters for Loss
        self.PATH = './DATASET/model2017-1_bfm_nomouth.h5'
        self.MAX_FEATURES = 500
        self.GOOD_MATCH_PERCENT = 0.15

        self.checkpoint_dir = "./DATASET/training/"
        self.checkpoint_path = "./DATASET/training/cp-{epoch:04d}.ckpt"

        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(
            self.checkpoint_path, verbose=1, save_weights_only=True,
            # Save weights, every 5-epochs.
            period=10)

        self.batch_stats_callback = batch_stats.CollectBatchStats()

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
        # base_model = tf.keras.applications.xception.Xception(include_top=False,
        #                                                      weights='imagenet',
        #                                                      input_tensor=None,
        #                                                      input_shape=self.IMG_SHAPE)
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

        # load trained weights from bootstrapping
        latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        model_o.load_weights(latest)

        return model_o

    def regularization_term(self, y_pred):
        shape = tf.slice(y_pred, begin=[0, 0], size=[self.BATCH_SIZE, 80], name='shape')
        expression = tf.slice(y_pred, begin=[0, 80], size=[self.BATCH_SIZE, 64], name='expression')
        reflectance = tf.slice(y_pred, begin=[0, 144], size=[self.BATCH_SIZE, 80], name='reflectance')

        shape2 = tf.math.square(shape, name='shape_squared')
        reflectance2 = tf.math.multiply(tf.math.square(reflectance), 1.7 * 10e-3)
        expression2 = tf.math.multiply(tf.math.square(expression), 0.8)

        regularization_term = tf.math.reduce_sum(shape2, axis=1) + tf.math.reduce_sum(reflectance2, axis=1) + \
            tf.math.reduce_sum(expression2, axis=1)

        regularization_term = tf.math.reduce_mean(regularization_term)  # shape = ()

        return regularization_term

    def loss_layer(self, y):
        loss = y

        return loss

    def model_loss(self):
        """" Wrapper function which calculates auxiliary values for the complete loss function.
         Returns a *function* which calculates the complete loss given only the input and target output """

        # Model space parameter loss
        regularization = self.regularization_term

        def custom_loss(y_true, y_pred):

            weight_reg = 2.9*10e-5
            # shape = tf.slice(y_pred, begin=[0, 0], size=[BATCH_SIZE, 80], name='shape')
            # expression = tf.slice(y_pred, begin=[0, 80], size=[BATCH_SIZE, 64], name='expression')
            # reflectance = tf.slice(y_pred, begin=[0, 144], size=[BATCH_SIZE, 80], name='reflectance')
            # rotation = tf.slice(y_pred, begin=[0, 224], size=[BATCH_SIZE, 3], name='rotation')
            # translation = tf.slice(y_pred, begin=[0, 227], size=[BATCH_SIZE, 3], name='translation')
            # illumination = tf.slice(y_pred, begin=[0, 230], size=[BATCH_SIZE, 27], name='illumination')

            y = tf.math.subtract(y_pred, y_true, name='pred_minus_true')
            print("ypred:", y_pred)
            # Model Space Parameter Loss
            reg_term = regularization(y_pred)
            model_loss = weight_reg * reg_term

            return model_loss

        return custom_loss

    def compile(self):
        """ Compiles the Keras model. Includes metrics to differentiate between the two main loss terms """
        self.model.compile(optimizer=tf.keras.optimizers.Adadelta(lr=self.BASE_LEARNING_RATE,
                                                                  rho=0.95, epsilon=None, decay=self.WEIGHT_DECAY),
                           loss=self.loss_func,
                           metrics=[tf.keras.losses.mean_squared_error, tf.keras.losses.mean_absolute_error])
        print('Model Compiled!')


def main():
    train = InverseFaceNet()

    train.compile()

main()