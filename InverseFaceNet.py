import tensorflow as tf
from keras import backend as K
import numpy as np
import random
import ImageFormationLayer as ifl
from tensorflow.keras import backend as K
import cv2


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
        """ Create a Keras model """

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

    @staticmethod
    def vector2dict(vector):
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

    def dense_photometric_alignment(self, x, original_image):
        # TODO original_image has to come from the tf.dataset
        formation = ifl.ImageFormationLayer(self.PATH, x)
        new_image = formation.get_reconstructed_image()
        # plt.imshow(new_image)
        # plt.show()
        # plt.imshow(original_image)
        # plt.show()

        new_image_aligned = self.align_images(new_image, original_image)

        # plt.imshow(new_image_aligned)
        # plt.show()

        # photo_term = sum(sum(np.linalg.norm(original_image - new_image, axis=2))) / 53149
        photo_term = sum(sum(np.linalg.norm(original_image - new_image_aligned, axis=2))) / 53149

        # print("photo term", photo_term)

        return photo_term

    def align_images(self, new_image, original_image):
        # Convert images to grayscale
        im1_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(self.MAX_FEATURES)
        keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * self.GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        # Draw top matches
        # imMatches = cv2.drawMatches(new_image, keypoints1, original_image, keypoints2, matches, None)
        # cv2.imwrite("matches.jpg", imMatches)

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        # Use homography
        height, width, channels = original_image.shape
        im1Reg = cv2.warpPerspective(new_image, h, (width, height))

        return im1Reg

    def model_loss(self):
        """" Wrapper function which calculates auxiliary values for the complete loss function.
         Returns a *function* which calculates the complete loss given only the input and target output """
        # Photometric alignment Loss
        # x = self.model.outputs
        # print("x: ", x)
        # print("shape x: ", x.shape)
        # original_image = self.model.inputs
        # print("im: ", original_image)
        # print("shape im: ", original_image.shape)

        # Photometric alignment Loss
        photo_loss_func = self.dense_photometric_alignment
        # Regularization Loss
        reg_loss_func = self.statistical_regularization_term

        def custom_loss(y_true, y_pred):
            x = y_pred
            x = K.transpose(x)
            x = self.vector2dict(x)

            original_image = self.model.input
            original_image = K.squeeze(original_image, 0)

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
