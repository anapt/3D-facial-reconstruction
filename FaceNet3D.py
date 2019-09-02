import numpy as np
import tensorflow as tf
from CollectBatchStats import CollectBatchStats


class FaceNet3D:
    def __init__(self):
        """
        Class initializer
        Parent class, includes all constants and paths used throughout the project
        """
        # path to Basel Model
        self.path = './DATASET/model2017-1_bfm_nomouth.h5'
        # number of vertices
        self.num_of_vertices = 53149
        # number of cells
        self.num_of_cells = 105694
        # number of variables for shape
        # can be increased up to 199
        self.shape_dim = 64
        # number of variables for expression
        # can be increased up to 100
        self.expression_dim = 64
        # number of variables for color
        # can be increased up to 199
        self.color_dim = 100
        # number of variables for rotation
        self.rotation_dim = 3
        # number of variables for translation
        self.translation_dim = 3
        # length of semantic code vector
        self.scv_length = self.shape_dim + self.expression_dim + self.color_dim + self.rotation_dim
        # path to save vectors
        # self.vector_path = "./DATASET/semantic/validation/x_{:06}.txt"
        self.vector_path = "./DATASET/bootstrapping/semantic/x_{:06}.txt"
        # path to save full patch
        self.no_crop_path = "./DATASET/images/no_crop/image_{:06}.png"
        # path to save cropped image
        # self.cropped_path = "./DATASET/images/validation/image_{:06}.png"
        self.cropped_path = "./DATASET/bootstrapping/images/image_{:06}.png"
        # if script is used for testing (aka images with no crop are needed) set variable to True
        self.testing = False
        # Landmark predictor path
        self.predictor_path = "./DATASET/shape_predictor_68_face_landmarks.dat"
        # specify whether in 'training' 'bootstrapping' or 'validation' phase
        self._case = 'bootstrapping'
        # dataset root folders path
        self.data_root = './DATASET/images/'
        self.sem_root = './DATASET/semantic/'
        self.bootstrapping_path = './DATASET/bootstrapping/'

        # DATASET AND NETWORK TRAINING OPTIONS
        self.IMG_SIZE = 224
        self.COLOR_CHANNELS = 3
        self.IMG_SHAPE = (self.IMG_SIZE, self.IMG_SIZE, self.COLOR_CHANNELS)

        self.WEIGHT_DECAY = 0.001
        # self.WEIGHT_DECAY = 0.000001
        self.BASE_LEARNING_RATE = 0.01

        self.BATCH_SIZE = 32
        self.BATCH_ITERATIONS = 7500

        self.SHUFFLE_BUFFER_SIZE = 10000

        self.checkpoint_dir = "./DATASET/training/"
        self.checkpoint_path = "./DATASET/training/cp-{epoch:04d}.ckpt"

        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(self.checkpoint_path, monitor='loss',
                                                              verbose=0, save_best_only=True,
                                                              save_weights_only=True, mode='min', save_freq='epoch')

        self.trained_models_dir = "./DATASET/trained_models/"

        self.batch_stats_callback = CollectBatchStats()

        # path to save training plots
        self.plot_path = './plots/'

    def vector2dict(self, vector):
        """
        Method that transforms (self.scv_length,) nd.array to dictionary

        :param vector: <class 'numpy.ndarray'> with shape (self.scv_length, ) : semantic code vector
        :return:
        dictionary with keys    shape           (self.shape_dim,)
                                expression      (self.expression_dim,)
                                color           (self.color_dim,)
                                rotation        (self.rotation_dim,)
        """
        if isinstance(vector, dict):
            return vector
        else:
            x = {
                "shape": np.squeeze(vector[0:self.shape_dim, ]),
                "expression": np.squeeze(vector[self.shape_dim:self.shape_dim+self.expression_dim, ]),
                "color": np.squeeze(vector[self.shape_dim+self.expression_dim:
                                           self.shape_dim+self.expression_dim+self.color_dim, ]),
                "rotation": np.squeeze(vector[self.shape_dim+self.expression_dim+self.color_dim:
                                              self.shape_dim+self.expression_dim+self.color_dim+self.rotation_dim, ])
            }
            return x

    def dict2vector(self, x):
        """
        Method that transforms (self.scv_length,) nd.array to dictionary

        :param x: dictionary with keys      shape           (self.shape_dim,)
                                            expression      (self.expression_dim,)
                                            color           (self.color_dim,)
                                            rotation        (self.rotation_dim,)
        :return: <class 'numpy.ndarray'> with shape (self.scv_length, ) : semantic code vector

        """
        vector = np.zeros(self.scv_length, dtype=float)
        vector[0:self.shape_dim, ] = x['shape']
        vector[self.shape_dim:self.shape_dim+self.expression_dim, ] = x['expression']
        vector[self.shape_dim+self.expression_dim:
               self.shape_dim+self.expression_dim+self.color_dim, ] = x['color']
        vector[self.shape_dim+self.expression_dim+self.color_dim:
               self.shape_dim+self.expression_dim+self.color_dim+self.rotation_dim, ] = x['rotation']

        return vector

    @staticmethod
    def translate(value, left_min, left_max, right_min=0, right_max=500):
        """
        Translates coordinates from range [left_min, left_max]
        to range [right_min, right_max]

        :param value:       value to translate
        :param left_min:    float
        :param left_max:    float
        :param right_min:   float
        :param right_max:   float
        :return: same shape and type as value
        """
        # Figure out how 'wide' each range is
        left_span = left_max - left_min
        right_span = right_max - right_min

        # Convert the left range into a 0-1 range (float)
        value_scaled = np.subtract(value, left_min) / float(left_span)

        # Convert the 0-1 range into a value in the right range.
        return right_min + (value_scaled * right_span)
